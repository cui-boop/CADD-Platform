from pathlib import Path
import random

import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    from rdkit import Chem

    test_mol = Chem.MolFromSmiles("CCO")
    RDKIT_AVAILABLE = test_mol is not None

except Exception:
    Chem = None
    RDKIT_AVAILABLE = False


try:
    from rdkit.Chem import Descriptors
except Exception:
    Descriptors = None

try:
    from rdkit.Chem import Crippen
except Exception:
    Crippen = None

try:
    from rdkit.Chem import Lipinski
except Exception:
    Lipinski = None

try:
    from rdkit.Chem import rdMolDescriptors
except Exception:
    rdMolDescriptors = None

try:
    from rdkit.Chem import Draw
except Exception:
    Draw = None


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"


def torch_available():
    return TORCH_AVAILABLE


def rdkit_available():
    try:
        from rdkit import Chem

        test_mol = Chem.MolFromSmiles("CCO")
        return test_mol is not None

    except Exception:
        return False


def canonicalize_smiles(smiles):
    """
    将 SMILES 标准化。
    如果 SMILES 无效，返回 None。
    """
    if smiles is None or str(smiles).strip() == "":
        return None

    smiles = str(smiles).strip()

    if not RDKIT_AVAILABLE:
        return smiles

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol)


def load_training_smiles(
    data_path="data/cleaned_activity.csv",
    smiles_col="smiles",
    label_col="label",
    active_only=True,
    max_len=120
):
    """
    从 cleaned_activity.csv 或其他 CSV 中读取训练用 SMILES。

    active_only=True 时，只使用 label=Active 的分子训练生成模型。
    这样模型会更倾向于学习活性分子的结构分布。
    """
    df = pd.read_csv(data_path)

    if smiles_col not in df.columns:
        raise ValueError(f"数据中缺少 {smiles_col} 列。")

    if active_only:
        if label_col not in df.columns:
            raise ValueError(f"选择 active_only=True 时，数据中必须包含 {label_col} 列。")
        df = df[df[label_col].astype(str).str.lower() == "active"]

    smiles_list = []

    for smi in df[smiles_col].dropna().astype(str).tolist():
        can = canonicalize_smiles(smi)

        if can is None:
            continue

        if len(can) <= max_len - 2:
            smiles_list.append(can)

    smiles_list = sorted(list(set(smiles_list)))

    if len(smiles_list) == 0:
        raise ValueError("没有可用于训练的有效 SMILES。")

    return smiles_list


def build_vocab(smiles_list):
    """
    基于训练 SMILES 构建字符表。
    """
    chars = sorted(list(set("".join(smiles_list))))

    itos = [PAD_TOKEN, START_TOKEN, END_TOKEN] + chars
    stoi = {ch: idx for idx, ch in enumerate(itos)}

    return stoi, itos


def encode_smiles(smiles, stoi, max_len=120):
    """
    将一个 SMILES 编码为定长 token id 序列。
    """
    tokens = [START_TOKEN] + list(smiles) + [END_TOKEN]

    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = END_TOKEN

    ids = [stoi[t] for t in tokens]

    pad_id = stoi[PAD_TOKEN]
    ids = ids + [pad_id] * (max_len - len(ids))

    return ids


if TORCH_AVAILABLE:
    class SmilesGRU(nn.Module):
        """
        轻量级字符级 SMILES-GRU 分子生成模型。
        输入前面的 SMILES 字符，预测下一个字符。
        """
        def __init__(
            self,
            vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embedding(x)
            output, hidden = self.gru(emb, hidden)
            logits = self.fc(output)
            return logits, hidden
else:
    class SmilesGRU:
        pass


def train_smiles_gru(
    smiles_list,
    model_path="models/smiles_gru_generator.pt",
    max_len=120,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    device="cpu",
    random_state=42
):
    """
    训练字符级 SMILES-GRU 生成模型。
    """
    if not TORCH_AVAILABLE:
        raise ImportError("当前环境未安装 PyTorch。请先安装 torch。")

    random.seed(random_state)
    torch.manual_seed(random_state)

    smiles_list = [
        s for s in sorted(list(set(smiles_list)))
        if isinstance(s, str) and len(s) <= max_len - 2
    ]

    if len(smiles_list) < 20:
        raise ValueError("有效训练 SMILES 少于 20 个，生成模型可能无法稳定训练。")

    stoi, itos = build_vocab(smiles_list)

    encoded = [
        encode_smiles(s, stoi, max_len=max_len)
        for s in smiles_list
    ]

    encoded = torch.tensor(encoded, dtype=torch.long)

    x = encoded[:, :-1]
    y = encoded[:, 1:]

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    device = torch.device(device)

    model = SmilesGRU(
        vocab_size=len(itos),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=stoi[PAD_TOKEN]
    )

    losses = []

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits, _ = model(xb)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        losses.append(
            {
                "epoch": epoch,
                "loss": avg_loss
            }
        )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": stoi,
        "itos": itos,
        "config": {
            "max_len": max_len,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "vocab_size": len(itos)
        },
        "train_smiles_set": list(smiles_list)
    }

    torch.save(checkpoint, model_path)

    train_info = {
        "training_smiles_count": len(smiles_list),
        "vocab_size": len(itos),
        "max_len": max_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "model_path": str(model_path)
    }

    return model_path, pd.DataFrame(losses), train_info


def load_generator(
    model_path="models/smiles_gru_generator.pt",
    device="cpu"
):
    """
    加载训练好的 SMILES-GRU 模型。
    """
    if not TORCH_AVAILABLE:
        raise ImportError("当前环境未安装 PyTorch。请先安装 torch。")

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    config = checkpoint["config"]

    model = SmilesGRU(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def sample_one_smiles(
    model,
    checkpoint,
    temperature=0.8,
    max_len=None,
    device="cpu"
):
    """
    使用训练好的模型采样生成一个 SMILES。
    """
    if not TORCH_AVAILABLE:
        raise ImportError("当前环境未安装 PyTorch。请先安装 torch。")

    stoi = checkpoint["vocab"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]

    if max_len is None:
        max_len = config["max_len"]

    device = torch.device(device)

    current_id = stoi[START_TOKEN]
    input_tensor = torch.tensor([[current_id]], dtype=torch.long).to(device)

    hidden = None
    generated_chars = []

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_tensor, hidden)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            next_token = itos[next_id]

            if next_token == END_TOKEN:
                break

            if next_token not in [PAD_TOKEN, START_TOKEN]:
                generated_chars.append(next_token)

            input_tensor = torch.tensor(
                [[next_id]],
                dtype=torch.long
            ).to(device)

    return "".join(generated_chars)


def calc_basic_properties(smiles):
    """
    计算生成分子的基础性质。
    """
    can = canonicalize_smiles(smiles)

    if can is None:
        return None

    if not rdkit_available():
        return {
            "smiles": smiles,
            "canonical_smiles": can,
            "valid_smiles": True
        }

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(can)

        if mol is None:
            return None

        return {
            "smiles": smiles,
            "canonical_smiles": can,
            "valid_smiles": True,
            "MolWt": round(Descriptors.MolWt(mol), 3),
            "LogP": round(Crippen.MolLogP(mol), 3),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
            "RingCount": int(Lipinski.RingCount(mol)),
            "AromaticRings": int(Lipinski.NumAromaticRings(mol)),
            "HeavyAtomCount": int(mol.GetNumHeavyAtoms())
        }

    except Exception:
        return None


def generate_smiles_table(
    model_path="models/smiles_gru_generator.pt",
    n_molecules=50,
    temperature=0.8,
    max_len=120,
    max_attempts=3000,
    filter_valid=True,
    filter_novel=True,
    device="cpu"
):
    """
    使用训练好的 SMILES-GRU 模型生成候选分子表。
    """
    model, checkpoint = load_generator(
        model_path=model_path,
        device=device
    )

    train_set = set(
        checkpoint.get("train_smiles_set", [])
    )

    rows = []
    seen = set()
    attempts = 0

    while len(rows) < n_molecules and attempts < max_attempts:
        attempts += 1

        raw_smi = sample_one_smiles(
            model=model,
            checkpoint=checkpoint,
            temperature=temperature,
            max_len=max_len,
            device=device
        )

        if raw_smi is None or raw_smi.strip() == "":
            continue

        props = calc_basic_properties(raw_smi)

        if props is None:
            if filter_valid:
                continue

            props = {
                "smiles": raw_smi,
                "canonical_smiles": "",
                "valid_smiles": False
            }

        canonical = props.get("canonical_smiles", raw_smi)

        if canonical in seen:
            continue

        if filter_novel and canonical in train_set:
            continue

        seen.add(canonical)

        rows.append(
            {
                "compound_id": f"GRU_GEN_{len(rows) + 1:04d}",
                "generation_method": "SMILES_GRU",
                "temperature": temperature,
                "novel": canonical not in train_set,
                **props
            }
        )

    summary = {
        "requested_molecules": n_molecules,
        "generated_molecules": len(rows),
        "attempts": attempts,
        "temperature": temperature,
        "filter_valid": filter_valid,
        "filter_novel": filter_novel
    }

    return pd.DataFrame(rows), summary


def save_generated_molecules(
    df,
    full_output_path="results/gru_generated_molecules.csv",
    compatible_output_path="results/generated_molecules.csv"
):
    """
    保存生成结果。
    full_output_path 保存完整结果；
    compatible_output_path 保存 compound_id, smiles，便于后续分子设计页面上传。
    """
    full_output_path = Path(full_output_path)
    compatible_output_path = Path(compatible_output_path)

    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    compatible_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        full_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    if df.empty:
        compatible_df = pd.DataFrame(
            columns=["compound_id", "smiles"]
        )
    else:
        smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        compatible_df = pd.DataFrame(
            {
                "compound_id": df["compound_id"],
                "smiles": df[smiles_col]
            }
        )

    compatible_df.to_csv(
        compatible_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return full_output_path, compatible_output_path


def get_mol_image(smiles, size=(250, 200)):
    """
    根据 SMILES 生成分子结构 SVG。

    返回 SVG 字符串；如果失败则返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return svg

    except Exception:
        return None
