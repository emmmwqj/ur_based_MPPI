import torch
import os
from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet

def test_model():
    # ---------------- 1. 配置路径与设备 ----------------
    dof = 6
    robot_name = "ur7e"
    model_path = os.path.expanduser(f"~/storm/weights/robot_self/{robot_name}_self_sdf.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}")
        return

    # ---------------- 2. 加载模型与权重 ----------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"📡 正在使用设备: {device}")

    nn_model = RobotSelfCollisionNet(n_joints=dof)
    checkpoint = torch.load(model_path, map_location=device)
    nn_model.model.load_state_dict(checkpoint['model_state_dict'])
    nn_model.model.to(device)
    nn_model.model.eval()

    # 获取归一化参数并移动到正确设备
    mean_x = checkpoint['norm']['x']['mean'].to(device)
    std_x = checkpoint['norm']['x']['std'].to(device)
    mean_y = checkpoint['norm']['y']['mean'].to(device)
    std_y = checkpoint['norm']['y']['std'].to(device)

    # ---------------- 3. 准备更具代表性的测试姿态 ----------------
    test_poses = {
        "绝对安全 (完全伸直)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 像一根旗杆立着，绝对不撞
        "已知碰撞 (手腕撞大臂)": [0.0, -1.57, 2.8, -1.57, 0.0, 0.0], # 手臂剧烈折叠
        "实际初始位姿": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0] # 你之前的"安全姿态"
    }
    
    for name, q_list in test_poses.items():
        # [关键修正 1] 将输入转为 Tensor 并增加 Batch 维度: [6] -> [1, 6]
        q_tensor = torch.tensor(q_list, dtype=torch.float32, device=device).unsqueeze(0)
        
        # [关键修正 2] 确保 mean 和 std 也能正确广播运算
        # (q_tensor 是 [1, 6], mean_x 如果是 [6], PyTorch 会自动处理)
        q_norm = (q_tensor - mean_x) / std_x
        
        # 模型推理
        with torch.no_grad():
            # 现在输入的维度是 [1, 6]，满足 torch.cat(..., 1) 的要求
            dist_pred_norm = nn_model.model(q_norm)
            
            # 反归一化得到真实物理距离 (米)
            dist_pred = dist_pred_norm * std_y + mean_y
            
        # 结果判定
        dist_m = dist_pred.item()
        status = "🔴 碰撞/危险" if dist_m < 0.02 else "🟢 安全"
        print(f"{name}:")
        print(f"  预测最小距离: {dist_m:.4f} 米 ({status})")

if __name__ == '__main__':
    test_model()
