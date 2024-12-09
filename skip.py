from time import sleep
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# 示例 Dataset
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 模拟一个加载失败的情况
        if idx % 5 == 0:  # 假设每 5 个数据加载失败
            raise ValueError(f"Failed to load data at index {idx}")
        return self.data[idx]

# 数据集和 DataLoader
dataset = ExampleDataset(list(range(100)))
dataloader = DataLoader(dataset, batch_size=1)

a = iter(dataloader)
# for x in a:
#     sleep(1)
with tqdm(total=len(dataloader),desc="Testing") as bar:
    while True:
        try:
            # print("retrieving")
            sleep(0.1)
            batch = next(a)
            print(batch)
        except ValueError as e:
            print(e)
            continue
        except StopIteration:
            # print("Iteration finished.")
            break
        finally:
            bar.update(1)
