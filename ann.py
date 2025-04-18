"""
Problem tanımı: mnist veri seti kullanılarak veri seti ile rakam sınıflandırma projesi
MNIST
ANN: Artificial Neural Networks
"""


#library 
import torch #tensor işlemleri
import torch.nn as nn # using for define ann layers
import torch.optim as optim # includes optimization algorithms
import torchvision #image processing and pre-defined models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#optional: cihazi belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#upload dataset
def get_data_loaders(batch_size=64):
    
   transform = transforms.Compose([transforms.ToTensor(),     #görüntüyü tensore çevirir ve 0-255->0-1 ölçeklendirir
                        transforms.Normalize((0.5,), (0.5,))
                        ]) 
   #mnist veri setini indir ve eğitim test kümelerini oluştur
   train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
   test_set = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
   
   #pytorch veri yükleyicisini oluştur
   train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
   
   return train_loader,test_loader

#train_loader,test_loader=get_data_loaders()

#data visualization
def visualize_samples(loader,n):
    images,labels=next(iter(loader))
    fig,axes = plt.subplots(1,n,figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray")
        axes[i].set_title(f"Label:{labels[i].item()}")
        axes[i].axis("off")
    plt.show()
        
        
#visualize_samples(train_loader,4)
        

#define ann model

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        
        self.flatten = nn.Flatten() #elimizde görüntüleri (2D) vektör haline çevirme(1D)
        
        self.fc1 = nn.Linear(28*28, 128) # ilk tam bağlı katman 28*28->input size, 128-> output
        self.relu = nn.ReLU() #activation func.
        
        self.fc2 = nn.Linear(128,64) #ikinci tam bağlı katman 128->input, 64->output
        
        self.fc3 = nn.Linear(64,10) #output layer, output sayısı veri setine göre değişken
        
        
    def forward(self,x):
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x


#create model and compile
#model = NeuralNetwork().to(device)

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(),lr=0.001)
    )

#criterion,optimizer = define_loss_and_optimizer(model)

#train
def train_model(model,train_loader,criterion,optimizer,epochs=10):
    
    #modeli eğitim moduna alalım
    model.train()
    #her bir epoch sonucunda elde edilen loss değerlerini saklamak için bir liste
    train_losses = []
    #belirtilen epoch sayısı kadar eğitim
    for epoch in range(epochs):
        total_loss = 0
        #tüm eğitim verileri üzerinde iterasyon gerçekleşir
        for images, labels in train_loader:
            images,labels = images.to(device),labels.to(device)
            #gradyanları sıfırla
            optimizer.zero_grad()
            #modeli uygula ->forward prop.
            predictions = model(images)
            #loss hesaplama
            loss = criterion(predictions,labels)
            #backprop. 
            loss.backward()
            #uptade weights
            optimizer.step()

            total_loss = total_loss + loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs},Loss: {avg_loss:.3f}")
    
    #loss graph
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker = "o",linestyle = "-",label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    

#train_model(model, train_loader, criterion, optimizer,epochs=5)
    

#test
def test_model(model,test_loader):
    model.eval() # modeli değerlendirme modu
    correct = 0 # doğru tahmin sayısı
    total = 0  # toplam veri sayacı
    
    with torch.no_grad(): # gradyan hesaplama olmadan
        for images, labels in test_loader: # test verisetini döngüye al 
            images, labels = images.to(device), labels.to(device) # cihaza taşı
            predictions = model(images) 
            _, predicted = torch.max(predictions,1) # en yüksek olasılıklı sınıfın etiketini bul
            total += labels.size(0) # toplam veri sayısı güncelle
            correct += (predicted==labels).sum().item() # doğru tahminleri say
            
    print(f"Test Accuracy: {100*correct/total:.3f}%")
    
    
#test_model(model, test_loader)
#%%
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader,5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




