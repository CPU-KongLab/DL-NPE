
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 数据归一化与标准化

# 图像标准化
transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],  # 取决于数据集
    std=[1, 1, 1])


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            #transforms.Resize(73),

            transforms.ToTensor(),
            transform_BZ
        ])
        self.val_tf = transforms.Compose([
            #transforms.Resize(73),
            transforms.ToTensor(),
            transform_BZ
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    # def padding_black(self, img):
    #     w, h  = img.size
    #     scale = 224. / max(w, h)
    #     img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    #     size_fg = img_fg.size
    #     size_bg = 224
    #     img_bg = Image.new("RGB", (size_bg, size_bg))
    #     img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
    #                           (size_bg - size_fg[1]) // 2))
    #     img = img_bg
    #     return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        #img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


def write_result(fileloc, epoch, trainloss, testloss, testaccuracy):
    with open(fileloc,"a") as f:
        data = "Epoch: "+ str(epoch) + "\tTrainLoss " + str(trainloss) + "\tTestLoss " + str(testloss)+ "\tTestAccuracy " + str(testaccuracy)+ "\n"
        f.write(data)


if __name__ == "__main__":
    train_dataset = LoadData("D:/Code_learning/SteroidXtract_codes&manuals_20210201/database/db/MS_figure/new/dataset/train.txt",
                             True)
    print("数据个数：", len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)


    for image, label in train_loader:
        print(image.shape)
        print(image)
        # img = transform_BZ(image)
        # print(img)
        print(label)

