import numpy as np
import matplotlib.pyplot as plt


## a - training loss
## b - training accuracy
## c - validation accuracy
## d - validation loss


### supervised_model_pretrained
a = np.load("numpy_files/a.npy")
b = np.load("numpy_files/b.npy")
c = np.load("numpy_files/c.npy")
d = np.load("numpy_files/d.npy")

### supervised_model_scratch
a1 = np.load("numpy_files/a1.npy")
b1 = np.load("numpy_files/b1.npy")
c1 = np.load("numpy_files/c1.npy")
d1 = np.load("numpy_files/d1.npy")

###  dino_model_finetuned
a_ = np.load("numpy_files/a_.npy")
b_ = np.load("numpy_files/b_.npy")
c_ = np.load("numpy_files/c_.npy")
d_ = np.load("numpy_files/d_.npy")

### dino_model_transfer_learning
a1_ = np.load("numpy_files/a1_.npy")
b1_ = np.load("numpy_files/b1_.npy")
c1_ = np.load("numpy_files/c1_.npy")
d1_ = np.load("numpy_files/d1_.npy")


## Moving Average Validation Accuracy of all models
avg_c = np.zeros_like(c)
avg_c[0] = c[0]
for i in range(1,len(c)):
  avg_c[i] = np.mean(c[max(0,i-10):i+1])

avg_c1 = np.zeros_like(c1)
avg_c1[0] = c1[0]
for i in range(1,len(c1)):
  avg_c1[i] = np.mean(c1[max(0,i-10):i+1])

avg_c_ = np.zeros_like(c_)
avg_c_[0] = c_[0]
for i in range(1,len(c_)):
  avg_c_[i] = np.mean(c_[max(0,i-10):i+1])

avg_c1_ = np.zeros_like(c1_)
avg_c1_[0] = c1_[0]
for i in range(1,len(c1_)):
  avg_c1_[i] = np.mean(c1_[max(0,i-10):i+1])

## plot
plt.figure()
plt.plot(avg_c,label = "supervised_model_pretrained",linewidth=4)
plt.plot(avg_c1,label = "supervised_model_scratch",linewidth=4)
plt.plot(avg_c_,label = "dino_model_finetuned",linewidth=4)
plt.plot(avg_c1_,label = "dino_model_transfer_learning",linewidth=4)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
# plt.title("Moving Average Validation Accuracy")
plt.savefig("plots/moving_average_validation_accuracy.png")
plt.show()


## Moving Average Validation Loss of all models
avg_d = np.zeros_like(d)
avg_d[0] = d[0]
for i in range(1,len(d)):
  avg_d[i] = np.mean(d[max(0,i-10):i+1])

avg_d1 = np.zeros_like(d1)
avg_d1[0] = d1[0]
for i in range(1,len(d1)):
  avg_d1[i] = np.mean(d1[max(0,i-10):i+1])

avg_d_ = np.zeros_like(d_)
avg_d_[0] = d_[0]
for i in range(1,len(d_)):
  avg_d_[i] = np.mean(d_[max(0,i-10):i+1])

avg_d1_ = np.zeros_like(d1_)
avg_d1_[0] = d1_[0]
for i in range(1,len(d1_)):
  avg_d1_[i] = np.mean(d1_[max(0,i-10):i+1])

# ## plot
# plt.figure()
# plt.plot(avg_d,label = "supervised_model_pretrained")
# plt.plot(avg_d1,label = "supervised_model_scratch")
# plt.plot(avg_d_,label = "dino_model_finetuned")
# plt.plot(avg_d1_,label = "dino_model_transfer_learning")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Validation Loss")
# plt.title("Moving Average Validation Loss")
# plt.savefig("plots/moving_average_validation_loss.png")
# plt.show()







## Moving Average Training Loss of all models
avg_a = np.zeros_like(a)
avg_a[0] = a[0]
for i in range(1,len(a)):
    avg_a[i] = np.mean(a[max(0,i-10):i+1])

avg_a1 = np.zeros_like(a1)
avg_a1[0] = a1[0]
for i in range(1,len(a1)):
    avg_a1[i] = np.mean(a1[max(0,i-10):i+1])

avg_a_ = np.zeros_like(a_)
avg_a_[0] = a_[0]
for i in range(1,len(a_)):
    avg_a_[i] = np.mean(a_[max(0,i-10):i+1])

avg_a1_ = np.zeros_like(a1_)
avg_a1_[0] = a1_[0]
for i in range(1,len(a1_)):
    avg_a1_[i] = np.mean(a1_[max(0,i-10):i+1])

## plot
plt.figure()
plt.plot(avg_a,label = "supervised_model_pretrained",linewidth=4)
plt.plot(avg_a1,label = "supervised_model_scratch",linewidth=4)
plt.plot(avg_a_,label = "dino_model_finetuned",linewidth=4)
plt.plot(avg_a1_,label = "dino_model_transfer_learning",linewidth=4)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
# plt.title("Moving Average Training Loss of all models")
plt.savefig("plots/moving_average_training_loss.png")
plt.show()

# ## Side by side plots of moving average training loss and validation accuracy
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(avg_a,label = "supervised_model_pretrained")
# plt.plot(avg_a1,label = "supervised_model_scratch")
# plt.plot(avg_a_,label = "dino_model_finetuned")
# plt.plot(avg_a1_,label = "dino_model_transfer_learning")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Training Loss")
# plt.title("Moving Average Training Loss")
# plt.subplot(1,2,2)
# plt.plot(c,label = "supervised_model_pretrained")
# plt.plot(c1,label = "supervised_model_scratch")
# plt.plot(c_,label = "dino_model_finetuned")
# plt.plot(c1_,label = "dino_model_transfer_learning")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Validation Accuracy")
# plt.title("Moving Average Validation Accuracy")
# plt.savefig("plots/side_by_side.png")
# plt.show()

# Side by side plots of moving average training loss and validation accuracy
plt.figure(figsize=(15,6))

# Plotting training loss
plt.subplot(1,2,1)
line1, = plt.plot(avg_a, linewidth=4)
line2, = plt.plot(avg_a1, linewidth=4)
line3, = plt.plot(avg_a_, linewidth=4)
line4, = plt.plot(avg_a1_, linewidth=4)

plt.xlabel("Epochs")
plt.ylabel("Average Training Loss")
# plt.title("Moving Average Training Loss")

# Plotting validation accuracy
plt.subplot(1,2,2)
plt.plot(c, linewidth=4)
plt.plot(c1, linewidth=4)
plt.plot(c_, linewidth=4)
plt.plot(c1_, linewidth=4)

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
# plt.title("Moving Average Validation Accuracy")

# Common legend for both plots
# Adjusting the location of the legend
plt.figlegend( [line1, line2, line3, line4], 
              labels = ["supervised model (pretrained)", "supervised model (scratch)", "dino model (finetuned)", "dino model (transfer_learning)"], 
              loc = "upper center", 
              ncol=4, 
              bbox_to_anchor=(0.2,0.5, 1, 0.5)
            #   bbox_to_anchor=(0.5, 1.1)  # adjusting the position. You can play with these values to move it.
             )
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)

plt.tight_layout()
plt.savefig("plots/side_by_side.png")
plt.show()




# ## Plot training loss of all models
# plt.figure()
# plt.plot(a,label = "supervised_model_pretrained")
# plt.plot(a1,label = "supervised_model_scratch")
# plt.plot(a_,label = "dino_model_finetuned")
# plt.plot(a1_,label = "dino_model_transfer_learning")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Training Loss")
# plt.title("Training Loss of all models")
# plt.savefig("plots/training_loss.png")
# plt.show()


# ## Plot validation accuracy of all models
plt.figure()
plt.plot(c,label = "supervised_model_pretrained",linewidth=4)
plt.plot(c1,label = "supervised_model_scratch",linewidth=4)
plt.plot(c_,label = "dino_model_finetuned",linewidth=4)
plt.plot(c1_,label = "dino_model_transfer_learning",linewidth=4)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
# plt.title("Validation Accuracy of all models")
plt.savefig("plots/validation_accuracy.png")
plt.show()


# ## Plot validation loss of all models
# plt.figure()
# plt.plot(d,label = "supervised_model_pretrained")
# plt.plot(d1,label = "supervised_model_scratch")
# plt.plot(d_,label = "dino_model_finetuned")
# plt.plot(d1_,label = "dino_model_transfer_learning")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss of all models")
# plt.savefig("plots/validation_loss.png")
# plt.show()