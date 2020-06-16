import re

import matplotlib.pyplot as plt
import numpy as np

regex = r'[_ ](?:(?:loss)|(?:accuracy)): (.*?)[ \n]'

def draw_loss_vs_accuracy_graph(file_name):
    text = open(file_name, 'r').read()
    matcher = re.compile(regex)
    match = matcher.findall(text)
    match = [round(float(x), 3) for x in match]
    indices = {'train_loss':0, 'train_c1_loss':1, 'train_c2_loss':2, 'train_fine_loss':3,
    'train_c1_acc':4, 'train_c2_acc':5, 'train_fine_acc':6, 
    'test_loss':7, 'test_c1_loss':8, 'test_c2_loss': 9, 'test_fine_loss': 10,
    'test_c1_acc':11, 'test_c2_acc':12, 'test_fine_acc':13}
    size = len(indices)
    # print(match)
    loss_legend_titles = ['train_loss', 'train_c1_loss', 'train_c2_loss', 'train_fine_loss',
    'test_loss', 'test_c1_loss', 'test_c2_loss', 'test_fine_loss']

    accuracy_legend_titles = ['train_c1_acc', 'train_c2_acc', 'train_fine_acc', 'test_c1_acc', 'test_c2_acc', 'test_fine_acc']
    
    model_type = 16
    # '''
    # for title in loss_legend_titles:
    maxi = -1
    # plt.ylim(0, 2.3)
    x_axis = np.arange(0, 60, step=1)
    plt.yticks(np.arange(0, 2.7, .5))
    for title in loss_legend_titles:
        ys = [x for x in match[indices[title]::size]]
        maxi = np.max(ys + [maxi])
        plt.plot(x_axis, ys)
    print(maxi)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"H-CNN VGG{model_type} Loss")
    plt.legend(loss_legend_titles)
    plt.show()

    plt.clf()
    # plt.cla()
    # plt.close()
    #'''

    maxi = -1
    # plt.ylim(0, 2.3)
    x_axis = np.arange(0, 60, step=1)
    plt.yticks(np.arange(0, 1.3, step=0.1))
    for title in accuracy_legend_titles:
        ys = [x for x in match[indices[title]::size]]
        maxi = np.max(ys + [maxi])
        # intys = [float(x) for x in ys]
        # print(ys)
        # print(intys)
        plt.plot(x_axis, ys)
        # input("enter to continue")
    # plt.plot(np.linspace(0, 2, num=60))
    print(maxi)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"H-CNN VGG{model_type} Accuracy")
    plt.legend(accuracy_legend_titles)
    plt.show()

def draw_loss_vs_accuracy_graph_base(file_name):
    text = open(file_name, 'r').read()
    matcher = re.compile(regex)
    match = matcher.findall(text)
    match = [round(float(x), 3) for x in match]
    indices = {'train_loss':0, 'train_acc':1, 'test_loss':2, 'test_acc':3}
    size = len(indices)
    # print(match)
    loss_legend_titles = ['train_loss', 'test_loss']

    accuracy_legend_titles = ['train_acc', 'test_acc']
    legend_titles = ['train', 'test']
    
    model_type = 16
    # '''
    # for title in loss_legend_titles:
    maxi = -1
    # plt.ylim(0, 2.3)
    x_axis = np.arange(0, 60, step=1)
    plt.yticks(np.arange(0, 2.7, .1))
    for title in loss_legend_titles:
        ys = [x for x in match[indices[title]::size]]
        maxi = np.max(ys + [maxi])
        plt.plot(x_axis, ys)
    print(maxi)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"VGG{model_type} Loss")
    plt.legend(legend_titles)
    # plt.show()
    
    # return
    plt.clf()
    # plt.cla()
    # plt.close()
    #'''

    maxi = -1
    # plt.ylim(0, 2.3)
    x_axis = np.arange(0, 60, step=1)
    plt.yticks(np.arange(0, 1.3, step=0.02))
    for title in accuracy_legend_titles:
        ys = [x for x in match[indices[title]::size]]
        maxi = np.max(ys + [maxi])
        # intys = [float(x) for x in ys]
        # print(ys)
        # print(intys)
        plt.plot(x_axis, ys)
        # input("enter to continue")
    # plt.plot(np.linspace(0, 2, num=60))
    print(maxi)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"VGG{model_type} Accuracy")
    plt.legend(legend_titles)
    plt.show()


def draw_confusion_matrix():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        

def main():
    # draw_loss_vs_accuracy_graph("vgg16hcnn.txt")
    # draw_loss_vs_accuracy_graph_base("vgg16base.txt")
    draw_confusion_matrix()

if __name__ == "__main__":
    main()

'''
vgg16hcnn

score is:  [0.501291498593241, 0.007671138271689415, 0.11302127689123154, 0.3800680935382843, 0.9979000091552734, 0.965499997138977, 0.9322999715805054]
['loss',
 'c1_predictions_cifar10_loss',
 'c2_predictions_cifar100_loss',
 'predictions_cifar100_loss',
 'c1_predictions_cifar10_accuracy',
 'c2_predictions_cifar100_accuracy',
 'predictions_cifar100_accuracy']

 vgg16base
 score is:  [0.4064862727693515, 0.9336000084877014]
['loss', 'accuracy']
array([[873,   0,  16,  14,   4,   1,  85,   0,   7,   0],
       [  1, 986,   1,   8,   2,   0,   0,   0,   2,   0],
       [ 15,   0, 908,   8,  31,   0,  38,   0,   0,   0],
       [ 11,   1,   8, 948,  15,   0,  17,   0,   0,   0],
       [  0,   1,  30,  20, 912,   0,  36,   0,   1,   0],
       [  0,   0,   0,   0,   0, 983,   0,  11,   1,   5],
       [ 88,   0,  45,  17,  59,   0, 787,   0,   4,   0],
       [  0,   0,   0,   0,   0,   2,   0, 986,   0,  12],
       [  1,   0,   0,   4,   2,   1,   2,   1, 989,   0],
       [  1,   0,   0,   0,   0,   5,   0,  30,   0, 964]])


'''