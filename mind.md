tensorboard --logdir=./output --port 6006 --bind_all
python train.py -s /home/ysc/3d-gaussian-splatting/data/zhanting/train
python trainmask.py -s  /home/ysc/3dgs-npy/data/zhanting/train --sh_degree 2 --iterations 50000 --densify_until_iter 30000 -r 1 
--data_device cpu
graph TD
    %% --- 定义样式 ---
    %% 修复说明：去除了不支持的 'rounded' 和 'shape' 属性，这些应由节点括号决定
    classDef inputNode fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef processNode fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C,rx:5,ry:5
    classDef poolNode fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#F57F17,stroke-dasharray: 5 5
    classDef lossNode fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C
    classDef fusionNode fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px,color:#1B5E20
    classDef weightNode fill:#E0E0E0,stroke:#616161,stroke-width:1px,color:#424242

    %% --- 输入层 ---
    subgraph Inputs [输入数据 Input Formulation]
        IP[渲染图<br>I_pred]:::inputNode
        IG[真值图<br>I_gt]:::inputNode
        M[有效性掩膜<br>Mask M]:::inputNode
    end

    %% --- Scale 0 (原始分辨率) ---
    subgraph Scale0 [Scale 0: 原始分辨率 H, W - 关注微观纹理]
        direction LR
        S0_SSIM(Masked SSIM 计算<br>1 - SSIM):::processNode
        S0_Loss(Loss 0: <br>L_ssim^0):::lossNode
    end

    IP --> S0_SSIM
    IG --> S0_SSIM
    M --> S0_SSIM
    S0_SSIM --> S0_Loss

    %% --- 下采样到 Scale 1 ---
    subgraph Downsample1 [下采样操作 Downsampling]
        Pool1_IP[2x2 Avg Pooling]:::poolNode
        Pool1_IG[2x2 Avg Pooling]:::poolNode
        Pool1_M[2x2 Avg Pooling]:::poolNode
    end

    IP -.-> Pool1_IP
    IG -.-> Pool1_IG
    M -.-> Pool1_M

    %% --- Scale 1 (中等分辨率) ---
    subgraph Scale1 [Scale 1: 中等分辨率 H/2, W/2 - 关注中频信息]
        direction LR
        IP1(I_pred^1):::inputNode
        IG1(I_gt^1):::inputNode
        M1(M^1):::inputNode
        S1_SSIM(Masked SSIM 计算<br>1 - SSIM):::processNode
        S1_Loss(Loss 1: <br>L_ssim^1):::lossNode
    end

    Pool1_IP --> IP1
    Pool1_IG --> IG1
    Pool1_M --> M1
    IP1 --> S1_SSIM
    IG1 --> S1_SSIM
    M1 --> S1_SSIM
    S1_SSIM --> S1_Loss

    %% --- 下采样到 Scale 2 ---
    subgraph Downsample2 [下采样操作 Downsampling]
        Pool2_IP[2x2 Avg Pooling]:::poolNode
        Pool2_IG[2x2 Avg Pooling]:::poolNode
        Pool2_M[2x2 Avg Pooling]:::poolNode
    end

    IP1 -.-> Pool2_IP
    IG1 -.-> Pool2_IG
    M1 -.-> Pool2_M

    %% --- Scale 2 (低分辨率) ---
    subgraph Scale2 [Scale 2: 低分辨率 H/4, W/4 - 关注宏观结构]
        direction LR
        IP2(I_pred^2):::inputNode
        IG2(I_gt^2):::inputNode
        M2(M^2):::inputNode
        S2_SSIM(Masked SSIM 计算<br>1 - SSIM):::processNode
        S2_Loss(Loss 2: <br>L_ssim^2):::lossNode
    end

    Pool2_IP --> IP2
    Pool2_IG --> IG2
    Pool2_M --> M2
    IP2 --> S2_SSIM
    IG2 --> S2_SSIM
    M2 --> S2_SSIM
    S2_SSIM --> S2_Loss

    %% --- 加权融合 ---
    subgraph Fusion [加权融合 Weighted Fusion]
        W0{{权重 w0}}:::weightNode
        W1{{权重 w1}}:::weightNode
        W2{{权重 w2}}:::weightNode
        FinalFusion[加权求和与归一化<br>Weighted Sum & Normalize]:::fusionNode
        FinalLoss(最终总损失<br>Total Loss):::lossNode
    end

    S0_Loss ==> FinalFusion
    S1_Loss ==> FinalFusion
    S2_Loss ==> FinalFusion

    W0 -.-> FinalFusion
    W1 -.-> FinalFusion
    W2 -.-> FinalFusion

    FinalFusion ==> FinalLoss

    %% --- 图例 ---
    subgraph Legend [图例 Legend]
        direction LR
        L1(图像/掩膜数据):::inputNode --- L2(计算步骤):::processNode --- L3(下采样/池化):::poolNode --- L4(损失值):::lossNode
    end