import wandb
import pandas

TARGET_COL = "asciiname"
START_CHAR = "<"
END_CHAR = ">"
PADDING_CHAR = "#"
#VOCAB = list(""" #',-.0123<>ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÅÇÉÖÚÜßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýĀāăąćČčĐēėęěğīİıŁłńňŌōŏőœřŚśŞşŠšŢţťūŭůźŻżŽžơưȘșȚțḑḨḩạấầịộ‘’""")
VOCAB = list(""" #'-.1<>ABCDEFGHIJKLMNOPQRSTUVWXYZ`abcdefghijklmnopqrstuvwxyz""")
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

def encode(x):
    return [char_to_idx[char] for char in x]

def decode(x):
    return "".join([idx_to_char[idx] for idx in x])

START_TOKEN = encode(START_CHAR)[0]
END_TOKEN = encode(END_CHAR)[0]
PADDING_TOKEN = encode(PADDING_CHAR)[0]

def prepare_dataset(df: pandas.DataFrame) -> pandas.DataFrame:
    df[TARGET_COL] = START_CHAR + df[TARGET_COL] + END_CHAR
    df["target_len"] = df[TARGET_COL].apply(len)

    max_len = max([len(city) for city in df[TARGET_COL].values])
    df[TARGET_COL] = df[TARGET_COL].str.pad(max_len, side="right", fillchar=PADDING_CHAR)

    chars = sorted(list(set("".join(df[TARGET_COL].values))))
    assert(VOCAB == chars)

    df["tokenized"] = df[TARGET_COL].apply(encode)

    return df

if __name__ == "__main__":
    # Initialize wandb
    run = wandb.init(project="woher", job_type="wherept-data")
    dataset = run.use_artifact("woher/cleaned-cities:latest").get("clean")
    df_raw = dataset.get_dataframe()

    df = prepare_dataset(df_raw)
    
    # Split
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    run.log({"vocab_len": len(VOCAB), "vocab": "".join(VOCAB), "train_size": len(train_df), "val_size": len(val_df)})
    
    artifact = wandb.Artifact(name='wherept-data', type='dataset')
    artifact.add(wandb.Table(dataframe=train_df), name="train_df")
    artifact.add(wandb.Table(dataframe=val_df), name="val_df")
    wandb.run.log_artifact(artifact)