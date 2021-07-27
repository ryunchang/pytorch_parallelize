import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.font_manager._rebuild()
from matplotlib import font_manager, rc
font_path = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import numpy as np
from caltech_class import label_tags 

columns = 6
rows = 6

def inference(testset):
    fig = plt.figure(figsize=(10,10))
    for i in range(1, columns*rows+1):
        print("--------------", i , "-------------")
        data_idx = np.random.randint(len(testset))
        # input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        #output = model(input_img)

        print("실제 : ", testset[data_idx][1])

        label = label_tags[testset[data_idx][1]]
        print("label : ", label)
        fig.add_subplot(rows, columns, i)
        # if pred == label:
        #     plt.title(pred + ', right !!')
        #     cmap = 'Blues'
        # else:
        #     plt.title('Not ' + pred + ' but ' +  label)
        #     cmap = 'Reds'
        plt.title(testset[data_idx][1])
        print(testset[data_idx])
        plot_img = testset[data_idx][0][0,:,:]
        plt.imshow(plot_img)
        plt.axis('off')
        print("-----------------------------")
    plt.show()      # If you want to measure inferencing time, comment out this line






transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

# datasets
testset = torchvision.datasets.Caltech101('../../data',
    download=True,
    transform=transform)

inference(testset)