#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# # Exploratory Data Analysis and Data Preprocessing

# In[4]:


df = pd.read_csv('smileannotationsfinal.csv',names=['id','text','category'])
df.set_index('id',inplace=True)


# In[5]:


df


# In[6]:


df.category.value_counts()


# In[7]:


df=df[~df.category.str.contains('\|')]


# In[8]:


df=df[df.category != 'nocode']


# In[9]:


df.category.value_counts()


# In[10]:


possible_labels=df.category.unique()


# In[11]:


possible_labels


# In[14]:


label_dict={}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label]=index


# In[15]:


label_dict


# In[17]:


df['label']=df.category.replace(label_dict)
df.tail()


# # Training / Validation Split

# In[18]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_val,y_train,y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size = 0.15,
    random_state = 131,
    stratify = df.label.values)


# In[21]:


df['data_type']= ['not set']*df.shape[0]


# In[22]:


df.loc[X_train,'data_type']='train'
df.loc[X_val,'data_type'] = 'val'


# In[23]:


df.head()


# In[24]:


df.groupby(['category','label','data_type']).count()


# # Loading Tokenizer and Encoding our Data using BERT Tokenizer and TensorDataset

# In[27]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[28]:


tokenizer=BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)


# In[30]:


encoded_data_train = tokenizer.batch_encode_plus(
df[df.data_type=='train'].text.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
df[df.data_type=='val'].text.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)


# In[31]:


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train= torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val= torch.tensor(df[df.data_type == 'val'].label.values)


# In[32]:


dataset_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
dataset_val = TensorDataset(input_ids_val,attention_masks_val,labels_val)


# In[34]:


len(dataset_train)


# In[35]:


len(dataset_val)


# # Setting up BERT pre trained Model

# In[37]:


from transformers import BertForSequenceClassification


# In[38]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(label_dict),
    output_attentions = False,
    output_hidden_states = False

)


# # Creating Data Loaders

# In[39]:


from torch.utils.data import DataLoader,RandomSampler,SequentialSampler


# In[40]:


dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size = 4
)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size = 32
)


# # Setting up Stochastic Optimizer and Scheduler

# In[41]:


from transformers import AdamW,get_linear_schedule_with_warmup


# In[42]:


optimizer =AdamW(
    model.parameters(),
    lr = 1e-5,
    eps=1e-8
)


# In[43]:


epochs = 10
scheduler= get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = len(dataloader_train)*epochs
)


# # Defining our Performance Metrics

# In[44]:


import numpy as np


# In[45]:


from sklearn.metrics import f1_score


# In[46]:


def f1_score_function(preds,labels):
    preds_flat = np.argmax(preds,axis =1).flatten()
    labels_flat = labels.flatten()
    return f1_score(preds_flat,labels_flat,average = 'weighted')


# In[47]:


def accuracy_per_class(preds,labels):
    label_dict_inverse = {v:k for k,v in label_dict.items()}
    preds_flat = np.argmax(preds,axis =1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


# # Creating our Training Loop

# In[49]:


import random
seed_value= 17
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


# In[50]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[51]:


device


# In[52]:


model.to(device)


# In[53]:


def evaluate(dataloader_val):
    model.eval()
    loss_val_total =0
    predictions,true_vals = [],[]
    
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':        batch[0],
                 'attention_mask':     batch[1],
                 'labels':      batch[2],}
        
        with torch.no_grad():
            outputs=model(**inputs)
            
        loss= outputs[0]
        logits = outputs[1]
        loss_val_total +=loss.item()
        
        logits=logits.detach().cpu().numpy()
        label_ids= inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions,axis =0)
    true_vals = np.concatenate(true_vals,axis =0)
    
    return loss_val_avg,predictions,true_vals
        


# In[56]:


for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train_total = 0 
    progress_bar= tqdm(dataloader_train,
                      desc = 'Epoch {:1d}'.format(epoch),
                      leave = False,
                      disable = False
                        )
    
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':        batch[0],
                 'attention_mask':     batch[1],
                 'labels':      batch[2]}
        outputs=model(**inputs)
        loss= outputs[0]
        loss_train_total +=loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss':  '{:.3f}'.format(loss.item()/len(batch))})
        
    torch.save(model.state_dict(),f'Models/BERT_ft_epoch{epoch}.model')
    tqdm.write('\nEpoch {epoch}')
    loss_train_avg= loss_train_total/len(dataloader)
    tqdm.write(f'Training Loss: {loss_train_avg}')
    
    val_loss,predictions,true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions,true_vals)
    tqdm.write(f'Validation Loss: {val_loss}')
    tqdm.write(f'F1 score: {val_f1}')
    
    
    
    
        
        


# # Loading and Evaluting our Model

# In[57]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(label_dict),
    output_attentions = False,
    output_hidden_states = False

)


# In[58]:


model.to(device)


# In[66]:


model.load_state_dict(torch.load('Models/BERT_ft_epoch.model',
                                map_location= torch.device('cpu')))


# In[67]:


_,predictions,true_vals = evaluate(dataloader_val)


# In[ ]:


accuracy_per_class(predictions,true_vals)


# In[ ]:




