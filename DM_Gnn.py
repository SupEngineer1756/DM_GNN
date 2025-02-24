import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import models, transforms
from openslide import OpenSlide
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

def load_wsi(file_path):
    print(f"Loading WSI: {file_path}")
    return OpenSlide(file_path)

def segment_tissue(slide, level=0, threshold=15):
    img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    return saturation > threshold

def adjust_dimensions(h, w, patch_size):
    h_adjusted = (h + patch_size - 1) // patch_size * patch_size
    w_adjusted = (w + patch_size - 1) // patch_size * patch_size
    return h_adjusted, w_adjusted
  
def crop_and_pad_mask(mask, patch_size):
    h, w = mask.shape
    h_adjusted, w_adjusted = adjust_dimensions(h, w, patch_size)
    padded_mask = np.zeros((h_adjusted, w_adjusted), dtype=mask.dtype)
    padded_mask[:h, :w] = mask
    return padded_mask

def crop_patches(slide, mask, patch_size=256, level=0):
    plt.imshow(mask)
    mask = crop_and_pad_mask(mask, patch_size)
    h, w = mask.shape
    scale_factor = int(slide.level_downsamples[level])
    print("h,w=",h,",",w)
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w , patch_size):
            patch_mask = mask[i:i+patch_size, j:j+patch_size]
            if patch_mask.mean() > 0.5:
                x_coord = int(j * scale_factor)
                y_coord = int(i * scale_factor)
                patch = np.array(
                    slide.read_region((x_coord, y_coord), level, (patch_size, patch_size))
                )[:, :, :3]
                patches.append(patch)
    return patches
def truncated_resnet50():
    # Load the pre-trained ResNet50 model
    full_model = models.resnet50(pretrained=True)
    
    # Extract layers up to the third residual block
    truncated_model = torch.nn.Sequential(
        full_model.conv1,    # Initial convolution block
        full_model.bn1,      # Batch normalization
        full_model.relu,     # Activation function
        full_model.maxpool,  # Max pooling
        
        full_model.layer1,   # First residual block
        full_model.layer2,   # Second residual block
        full_model.layer3    # Third residual block
    )
    
    # Set the model to evaluation mode (important for feature extraction)
    truncated_model.eval()
    
    return full_model

def extract_features(patches):
    # Load ResNet50 and truncate it to the first three residual blocks
    
    resnet50 = models.resnet50(pretrained=True)
    truncated_resnet = torch.nn.Sequential(
        resnet50.conv1,    # Initial Convolutional Layer
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,   # First Residual Block
        resnet50.layer2,   # Second Residual Block
        resnet50.layer3    # Third Residual Block
    )
    
    # Set the model to evaluation mode
    model = truncated_resnet
    model.eval()

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Ensure patches are resized correctly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for patch in patches:
            # Preprocess the patch and prepare input tensor
            input_tensor = transform(patch).unsqueeze(0)

            # Forward pass through the truncated ResNet
            feature_map = model(input_tensor)

            # Apply global average pooling
            pooled_feature = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))

            # Reduce to 1024 dimensions using a 1x1 convolution
            reduce_conv = torch.nn.Conv2d(1024, 1024, kernel_size=1)
            reduced_feature = reduce_conv(pooled_feature)

            # Flatten the pooled feature vector
            feature_vector = reduced_feature.view(reduced_feature.size(0), -1)
            #print("feature_size=", feature_vector.shape)
            # Add to the list of features
            features.append(feature_vector.cpu().numpy())

    # Convert features to a numpy array and return
    return np.vstack(features)

def process_wsi(file_path, patch_size=256, threshold=15, level=0):
    slide = load_wsi(file_path)
    mask = segment_tissue(slide, level, threshold)
    patches = crop_patches(slide, mask, patch_size, level)
    return extract_features(patches)

class DeepProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepProjectionLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.ln(x)

class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, features, adjacency_matrix):
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        degree_matrix_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix))
        normalized_adjacency = degree_matrix_inv_sqrt @ adjacency_matrix @ degree_matrix_inv_sqrt
        return F.relu(normalized_adjacency @ features @ self.weight)

class FeatureUpdatingBranch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureUpdatingBranch, self).__init__()
        self.dpl = DeepProjectionLayer(input_dim, hidden_dim)
        self.gcn1 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolutionLayer(hidden_dim, output_dim)
        self.ln = torch.nn.LayerNorm(output_dim)
        
    def forward(self, features, adjacency_matrix):
        features = self.dpl(features)
        features = self.gcn1(features, adjacency_matrix)
        features = self.gcn2(features, adjacency_matrix)
        features = self.gcn3(features, adjacency_matrix)
        features = self.ln(features)
        return features

def compute_affinity_matrix(features, threshold=0.5):
    similarity_matrix = cosine_similarity(features)
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()
    normalized_similarity = (similarity_matrix - min_val) / (max_val - min_val)
    affinity_matrix = (normalized_similarity > threshold).astype(float)
    return torch.tensor(affinity_matrix, dtype=torch.float32)

class GlobalAnalysisBranch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.3):
        super(GlobalAnalysisBranch, self).__init__()
        self.alpha = alpha
        self.dpl = DeepProjectionLayer(input_dim, hidden_dim)
        self.gcn1 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolutionLayer(hidden_dim, output_dim)
        self.ln = torch.nn.LayerNorm(output_dim)

    def compute_attention_weights(self, features):
        W1 = torch.nn.Linear(features.shape[1], features.shape[1])
        W2 = torch.nn.Linear(features.shape[1], features.shape[1])
        W3 = torch.nn.Linear(features.shape[1], 1)

        tanh_output = torch.tanh(W2(features))
        sigmoid_output = torch.sigmoid(W1(features))
        attention_weights = W3(sigmoid_output * tanh_output)
        return torch.sigmoid(attention_weights)

    def compute_mam(self, attention_weights):
        mam = self.alpha + (1 - self.alpha) * attention_weights @ attention_weights.T
        return mam

    def compute_ctm(self, spatial_distances, threshold=0.5):
        ctm = (spatial_distances < threshold).float()
        return ctm

    def forward(self, features, spatial_distances):
        features = self.dpl(features)

        # Attention Weights and CAM
        attention_weights = self.compute_attention_weights(features)
        mam = self.compute_mam(attention_weights)
        ctm = self.compute_ctm(spatial_distances)
        cam = mam + ctm

        # Graph Convolutional Network
        features = self.gcn1(features, cam)
        features = self.gcn2(features, cam)
        features = self.gcn3(features, cam)
        features = self.ln(features)
        return features

# Custom Dataset for WSIs and Clinical Data
class WSIDataset(Dataset):
    def __init__(self, svs_dir, clinical_csv, patch_size=256, level=0):
        self.svs_dir = svs_dir
        self.clinical_data = pd.read_csv(clinical_csv)
        self.patch_size = patch_size
        self.level = level
        self.svs_files = [
            os.path.join(svs_dir, f) for f in os.listdir(svs_dir) if f.endswith(".SVS")
        ]
        self.clinical_data.set_index("submitter_id", inplace=True)

    def __len__(self):
        return len(self.svs_files)

    def __getitem__(self, idx):
        svs_path = self.svs_files[idx]
        slide_id = os.path.basename(svs_path).replace(".SVS", "")
        survival_bin = self.clinical_data.loc[slide_id, "survival_bin"]
        event_indicator = self.clinical_data.loc[slide_id, "event_indicator"]
        return svs_path, survival_bin, event_indicator

# Affinity-Guided Attention Recalibration Module (AARM)
class AttentionRecalibrationModule(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(AttentionRecalibrationModule, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, afm, attention_weights, fub_features, gab_features):
        recalibrated_weights = afm @ attention_weights
        recalibrated_weights = recalibrated_weights / torch.sqrt(
            torch.sum(recalibrated_weights**2, dim=0, keepdim=True)
        )
        final_weights = self.alpha * recalibrated_weights + self.beta * attention_weights
        final_weights = torch.softmax(final_weights, dim=0)
        combined_features = torch.cat([fub_features, gab_features], dim=1)
        graph_representation = torch.sum(final_weights * combined_features, dim=0, keepdim=True)
        return graph_representation

# Loss Function (Cox Proportional Hazard Loss)
class CoxLoss(torch.nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, F, y, c, bins):
        hazard_weights = torch.nn.Linear(F.size(1), len(bins) - 1)(F)
        hazard_probs = torch.sigmoid(hazard_weights)
        survival_probs = torch.cumprod(1 - hazard_probs, dim=1)
        survival_at_y = torch.gather(survival_probs, dim=1, index=y.unsqueeze(1)).squeeze(1)
        survival_at_y_minus_1 = torch.gather(
            survival_probs, dim=1, index=(y - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        loss = -c * torch.log(survival_at_y + 1e-8) - (1 - c) * torch.log(
            survival_at_y_minus_1 + 1e-8
        )
        return loss.mean()



def compute_c_index(predictions, survival_times, event_indicators):
    """Compute the Concordance Index (c-index)."""
    c_index = concordance_index(survival_times, predictions, event_observed=event_indicators)
    return c_index

def plot_kaplan_meier(groups, survival_times, event_indicators, labels):
    """Plot Kaplan-Meier curves."""
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for group, label in zip(groups, labels):
        kmf.fit(survival_times[group], event_observed=event_indicators[group], label=label)
        kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curves")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_log_rank_test(group1_indices, group2_indices, survival_times, event_indicators):
    """Perform the log-rank test between two groups."""
    results = logrank_test(
        survival_times[group1_indices],
        survival_times[group2_indices],
        event_observed_A=event_indicators[group1_indices],
        event_observed_B=event_indicators[group2_indices],
    )
    return results.p_value


# Model Training and Testing
def train_and_test(svs_dir, clinical_csv, bins, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WSIDataset(svs_dir, clinical_csv)
    
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    trainingdataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    testingdataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    instances=[]

    for svs_path, survival_bin, event_indicator in trainingdataloader:
      instance=[]

      features = process_wsi(svs_path[0])
      features_tensor = torch.tensor(features, dtype=torch.float32)

      survival_bin = survival_bin
      event_indicator = event_indicator

      instance.append(features)
      instance.append(features_tensor)
      instance.append(survival_bin)
      instance.append(event_indicator)

      instances.append(instance)
      
    print("Features extracted!")

    fub = FeatureUpdatingBranch(input_dim=1024, hidden_dim=512, output_dim=256)
    gab = GlobalAnalysisBranch(input_dim=1024, hidden_dim=512, output_dim=256)
    
    aarm = AttentionRecalibrationModule()
    loss_fn = CoxLoss()
    optimizer = torch.optim.Adam(list(fub.parameters()) + list(gab.parameters()) + list(aarm.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for instance in instances:
            # Compute adjacency matrices and spatial distances
            #Reextract features from the instance
            features=instance[0]
            features_tensor=instance[1]
            survival_bin=instance[2]
            event_indicator=instance[3]

            afm = compute_affinity_matrix(features)
            spatial_distances = torch.rand(features_tensor.size(0), features_tensor.size(0))

            # Parallel feedforward through FUB and GAB
            fub_output = fub(features_tensor, afm)
            gab_output = gab(features_tensor, spatial_distances)

            # Attention recalibration and graph representation
            attention_weights = gab.compute_attention_weights(features_tensor)
            graph_representation = aarm(afm, attention_weights, fub_output, gab_output)

            # Compute loss
            loss = loss_fn(graph_representation, survival_bin, event_indicator, bins)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    print("Training complete.")

    
    fub.eval()
    gab.eval()
    aarm.eval()

    survival_times = []
    event_indicators = []
    predictions = []

    testing_instances=[]

    for svs_path, survival_bin, event_indicator in testingdataloader:
      testing_instance=[]

      features = process_wsi(svs_path[0])
      features_tensor = torch.tensor(features, dtype=torch.float32)

      survival_bin = survival_bin
      event_indicator = event_indicator

      testing_instance.append(features)
      testing_instance.append(features_tensor)
      testing_instance.append(survival_bin)
      testing_instance.append(event_indicator)

      testing_instances.append(testing_instance)
    with torch.no_grad():
        for instance in testing_instances:
            # Compute adjacency matrices and spatial distances
            #Reextract features from the instance
            features=instance[0]
            features_tensor=instance[1]
            survival_bin=instance[2]
            event_indicator=instance[3]

            afm = compute_affinity_matrix(features)
            spatial_distances = torch.rand(features_tensor.size(0), features_tensor.size(0))

            fub_output = fub(features_tensor, afm)
            gab_output = gab(features_tensor, spatial_distances)
            attention_weights = gab.compute_attention_weights(features_tensor)

            risk_score = aarm(afm, attention_weights, fub_output, gab_output).mean().item()

            survival_times.append(survival_bin)
            event_indicators.append(event_indicator)
            predictions.append(risk_score)

    survival_times = np.array(survival_times)
    event_indicators = np.array(event_indicators)
    predictions = np.array(predictions)

    # Compute c-index
    c_index = compute_c_index(predictions, survival_times, event_indicators)
    print(f"C-Index: {c_index:.3f}")

    # Kaplan-Meier curves
    high_risk_group = predictions > np.median(predictions)
    low_risk_group = ~high_risk_group

    plot_kaplan_meier(
        [high_risk_group, low_risk_group],
        survival_times,
        event_indicators,
        labels=["High Risk", "Low Risk"]
    )

    # Log-rank test
    p_value = evaluate_log_rank_test(high_risk_group, low_risk_group, survival_times, event_indicators)
    print(f"Log-Rank Test p-value: {p_value:.3e}")



if __name__ == "__main__":
    svs_dir = "data2/"
    clinical_csv = "data2/DATA_LABELS_50.csv"
    bins = [0, 365, 730, 1095, float("inf")]

    train_and_test(svs_dir, clinical_csv, bins, epochs=20, lr=1e-3)






