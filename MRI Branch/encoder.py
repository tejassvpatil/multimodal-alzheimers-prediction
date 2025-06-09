import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        return 0.7 * self.mse(pred, target) + 0.3 * self.l1(pred, target)

class MRIDataset(Dataset):
    def __init__(self, data_dir, csv_path=None):
        self.data_dir = data_dir
        self.npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        self.rids = [f.replace('.npy', '') for f in self.npy_files]
        
        print(f"Found {len(self.npy_files)} MRI files")
    
    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        npy_path = os.path.join(self.data_dir, self.npy_files[idx])
        volume = np.load(npy_path)
        
        # Add channel dimension: (H, W, D) -> (1, H, W, D)
        volume = torch.FloatTensor(volume).unsqueeze(0)
        
        rid = self.rids[idx]
        
        return volume, rid

class MRI3DCNN(nn.Module):
    def __init__(self, input_shape=(91, 109, 91), embedding_dim=128):
        super(MRI3DCNN, self).__init__()
        
        self.spatial_dropout = nn.Dropout3d(0.2)
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),  # GroupNorm instead of BatchNorm for small batches
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (91,109,91) -> (45,54,45)
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (45,54,45) -> (22,27,22)
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (22,27,22) -> (11,13,11)
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (11,13,11) -> (5,6,5)
        )
        
        # Calculate flattened size
        self.flattened_size = 256 * 5 * 6 * 5  # 38400
        
        self.encoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.flattened_size),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_reshape = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1)),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1)),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1)),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=(1,1,1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        features = self.spatial_dropout(features) 
        flattened = features.view(features.size(0), -1)
        z_mri = self.encoder(flattened)
        
        mask_mri = torch.ones(x.size(0), dtype=torch.float32, device=x.device)
        
        return z_mri, mask_mri
    
    def encode(self, x):
        features = self.features(x)
        features = self.spatial_dropout(features)  
        flattened = features.view(features.size(0), -1)
        z_mri = self.encoder(flattened)
        return z_mri
    
    def decode(self, z_mri, target_shape=(91, 109, 91)):
        decoded_features = self.decoder(z_mri)
        decoded_features = decoded_features.view(-1, 256, 5, 6, 5)
        reconstructed = self.decoder_reshape(decoded_features)
        
        if reconstructed.shape[2:] != target_shape:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, 
                size=target_shape, 
                mode='trilinear', 
                align_corners=False
            )
        
        return reconstructed

def train_autoencoder_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)
        
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        optimizer.zero_grad()
        
        with autocast():
            z_mri = model.encode(data_normalized)
            reconstructed = model.decode(z_mri)
            loss = criterion(reconstructed, data_normalized)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def calculate_ssim(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0):
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    mu1 = np.mean(img1_flat)
    mu2 = np.mean(img2_flat)
    var1 = np.var(img1_flat)
    var2 = np.var(img2_flat)
    cov = np.mean((img1_flat - mu1) * (img2_flat - mu2))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    
    return ssim

def calculate_psnr(img1, img2, data_range=1.0):
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_mse = []
    all_rmse = []
    all_ssim = []
    all_psnr = []
    all_corr = []
    
    print("Evaluating model with quantitative metrics...")
    
    with torch.no_grad():
        for data, rids in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)
            
            data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
            z_mri = model.encode(data_normalized)
            reconstructed = model.decode(z_mri)
            for i in range(data.shape[0]):
                orig = data_normalized[i].squeeze().cpu().numpy()
                recon = reconstructed[i].squeeze().cpu().numpy()
                mse = mean_squared_error(orig.flatten(), recon.flatten())
                rmse = np.sqrt(mse)
                all_mse.append(mse)
                all_rmse.append(rmse)
                ssim = calculate_ssim(orig, recon)
                all_ssim.append(ssim)
                psnr = calculate_psnr(orig, recon)
                all_psnr.append(psnr)
                corr, _ = pearsonr(orig.flatten(), recon.flatten())
                all_corr.append(corr)
    metrics = {
        'MSE': np.mean(all_mse),
        'RMSE': np.mean(all_rmse),
        'SSIM': np.mean(all_ssim),
        'PSNR': np.mean(all_psnr),
        'Correlation': np.mean(all_corr),
        'MSE_std': np.std(all_mse),
        'RMSE_std': np.std(all_rmse),
        'SSIM_std': np.std(all_ssim),
        'PSNR_std': np.std(all_psnr),
        'Correlation_std': np.std(all_corr)
    }
    
    return metrics, (all_mse, all_rmse, all_ssim, all_psnr, all_corr)

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings_dict = {}
    
    with torch.no_grad():
        for data, rids in tqdm(dataloader, desc="Extracting embeddings"):
            data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
            data_normalized = data_normalized.to(device)
            z_mri = model.encode(data_normalized)
            for i, rid in enumerate(rids):
                embeddings_dict[rid] = z_mri[i].cpu().numpy()
    
    return embeddings_dict

def plot_training_curves(train_losses, learning_rates, save_path='plots/training_curves.png'):
    os.makedirs('plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(learning_rates, 'r-', linewidth=2, label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_evaluation_metrics(metrics, metric_distributions, save_path='plots/evaluation_metrics.png'):
    os.makedirs('plots', exist_ok=True)
    
    all_mse, all_rmse, all_ssim, all_psnr, all_corr = metric_distributions
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metric_names = ['MSE', 'RMSE', 'SSIM', 'PSNR', 'Correlation']
    metric_values = [metrics['MSE'], metrics['RMSE'], metrics['SSIM'], metrics['PSNR'], metrics['Correlation']]
    metric_stds = [metrics['MSE_std'], metrics['RMSE_std'], metrics['SSIM_std'], metrics['PSNR_std'], metrics['Correlation_std']]
    
    ax = axes[0, 0]
    bars = ax.bar(metric_names, metric_values, yerr=metric_stds, capsize=5, alpha=0.7)
    ax.set_title('Average Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value')
    ax.tick_params(axis='x', rotation=45)
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    distributions = [all_mse, all_rmse, all_ssim, all_psnr, all_corr]
    titles = ['MSE Distribution', 'RMSE Distribution', 'SSIM Distribution', 'PSNR Distribution', 'Correlation Distribution']
    
    for i, (dist, title) in enumerate(zip(distributions, titles)):
        row, col = divmod(i + 1, 3)
        ax = axes[row, col]
        ax.hist(dist, bins=20, alpha=0.7, color=colors[i], density=True)
        ax.axvline(np.mean(dist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dist):.4f}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation metrics plot saved to {save_path}")

def plot_embedding_analysis(embeddings_dict, save_path='plots/embedding_analysis.png'):
    os.makedirs('plots', exist_ok=True)
    
    rids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[rid] for rid in rids])
    
    print(f"Analyzing {len(rids)} embeddings of dimension {embeddings.shape[1]}")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    pca_50 = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca_50.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(rids)//4))
    tsne_result = tsne.fit_transform(embeddings_pca)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50)
    ax.set_title(f'PCA of MRI Embeddings\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2%})', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=50)
    ax.set_title('t-SNE of MRI Embeddings', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    pca_full = PCA()
    pca_full.fit(embeddings)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 1]
    embedding_stats = {
        'Mean': np.mean(embeddings, axis=0),
        'Std': np.std(embeddings, axis=0),
        'Min': np.min(embeddings, axis=0),
        'Max': np.max(embeddings, axis=0)
    }
    
    for i, (stat_name, stat_values) in enumerate(embedding_stats.items()):
        ax.plot(stat_values, label=stat_name, linewidth=2)
    
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Embedding Statistics Across Dimensions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Embedding analysis plot saved to {save_path}")

def plot_reconstruction_samples(model, dataloader, device, n_samples=4, save_path='plots/reconstruction_samples.png'):
    os.makedirs('plots', exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for data, rids in dataloader:
            data = data.to(device)
            data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            z_mri = model.encode(data_normalized)
            reconstructed = model.decode(z_mri)
            
            mid_slice = data.shape[4] // 2 
            
            fig, axes = plt.subplots(2, min(n_samples, data.shape[0]), figsize=(4*min(n_samples, data.shape[0]), 8))
            
            for i in range(min(n_samples, data.shape[0])):
                orig_slice = data_normalized[i, 0, :, :, mid_slice].cpu().numpy()
                if n_samples > 1:
                    axes[0, i].imshow(orig_slice, cmap='gray')
                    axes[0, i].set_title(f'Original {rids[i]}')
                    axes[0, i].axis('off')
                else:
                    axes[0].imshow(orig_slice, cmap='gray')
                    axes[0].set_title(f'Original {rids[i]}')
                    axes[0].axis('off')
                recon_slice = reconstructed[i, 0, :, :, mid_slice].cpu().numpy()
                if n_samples > 1:
                    axes[1, i].imshow(recon_slice, cmap='gray')
                    axes[1, i].set_title(f'Reconstructed {rids[i]}')
                    axes[1, i].axis('off')
                else:
                    axes[1].imshow(recon_slice, cmap='gray')
                    axes[1].set_title(f'Reconstructed {rids[i]}')
                    axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Reconstruction samples saved to {save_path}")
            break 

def main():
    DATA_DIR = "mri_preproc"
    BATCH_SIZE = 12          
    LEARNING_RATE = 0.0005  
    NUM_EPOCHS = 250        
    EMBEDDING_DIM = 128    
    
    patience = 30
    best_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    learning_rates = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = MRIDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = MRI3DCNN(embedding_dim=EMBEDDING_DIM).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
   
    criterion = CombinedLoss()  
    optimizer = optim.AdamW(  
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_autoencoder_epoch(model, dataloader, criterion, optimizer, scaler, device)
        
        train_losses.append(avg_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/mri_best.ckpt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'weights/mri_checkpoint_{epoch+1}.ckpt')
    plot_training_curves(train_losses, learning_rates)
    torch.save(model.state_dict(), 'weights/mri_final.ckpt')
    print("\nEvaluating model performance...")
    metrics, metric_distributions = evaluate_model(model, dataloader, device)
    
    print(f"\n{'='*50}")
    print("MODEL EVALUATION METRICS")
    print(f"{'='*50}")
    print(f"MSE:          {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}")
    print(f"RMSE:         {metrics['RMSE']:.6f} ± {metrics['RMSE_std']:.6f}")
    print(f"SSIM:         {metrics['SSIM']:.4f} ± {metrics['SSIM_std']:.4f}")
    print(f"PSNR:         {metrics['PSNR']:.2f} ± {metrics['PSNR_std']:.2f} dB")
    print(f"Correlation:  {metrics['Correlation']:.4f} ± {metrics['Correlation_std']:.4f}")
    print(f"{'='*50}")
    
    plot_evaluation_metrics(metrics, metric_distributions)
    
    plot_reconstruction_samples(model, dataloader, device)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('evaluation_metrics.csv', index=False)
    print("Metrics saved to evaluation_metrics.csv")
    
    print("Extracting embeddings...")
    embeddings_dict = extract_embeddings(model, dataloader, device)
    plot_embedding_analysis(embeddings_dict)
    os.makedirs('embeddings/mri', exist_ok=True)
    embedding_paths = []
    
    for rid, z_mri in embeddings_dict.items():
        embedding_path = f'embeddings/mri/{rid}.npy'
        np.save(embedding_path, z_mri)
        embedding_paths.append({'RID': rid, 'path': embedding_path, 'mask': 1})

    embeddings_df = pd.DataFrame(embedding_paths)
    embeddings_df.to_csv('embeddings/mri_embeddings.csv', index=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    summary_text = f"""
    MRI Autoencoder Training Summary
    
    Dataset: {len(dataset)} MRI volumes
    Model Parameters: {sum(p.numel() for p in model.parameters()):,}
    Training Epochs: {len(train_losses)}
    Final Loss: {train_losses[-1]:.6f}
    Best Loss: {best_loss:.6f}
    
    Evaluation Metrics:
    • MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}
    • RMSE: {metrics['RMSE']:.6f} ± {metrics['RMSE_std']:.6f}
    • SSIM: {metrics['SSIM']:.4f} ± {metrics['SSIM_std']:.4f}
    • PSNR: {metrics['PSNR']:.2f} ± {metrics['PSNR_std']:.2f} dB
    • Correlation: {metrics['Correlation']:.4f} ± {metrics['Correlation_std']:.4f}
    
    Embedding Details:
    • Dimension: {EMBEDDING_DIM}
    • Count: {len(embeddings_dict)}
    • Saved to: embeddings/mri/
    
    Training Time: {(time.time() - start_time)/60:.2f} minutes
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Training Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Saved {len(embeddings_dict)} embeddings to embeddings/mri/")
    print(f"Model saved to weights/mri_best.ckpt")
    print(f"Each embedding is a 128-D array (z_mri). Mask (1 for existing, 0 for missing) handled at fusion level.")
    print(f"\nGenerated plots:")
    print(f"• Training curves: plots/training_curves.png")
    print(f"• Evaluation metrics: plots/evaluation_metrics.png")
    print(f"• Embedding analysis: plots/embedding_analysis.png")
    print(f"• Reconstruction samples: plots/reconstruction_samples.png")
    print(f"• Training summary: plots/training_summary.png")

if __name__ == "__main__":
    main()