#from Structures.StereotypeCorpus import StereotypeCorpus
# from load_data import get_corpus
from auxiliarFunction import TaskStereotype
from data_deception import d_name_f
from utils import get_model_and_tokenizer, save_matrices, save_attentions
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from trM import print_conf_matrix, conf_examples, get_mostfrequent
import os
from transf_text import apply_mask
from bertviz import head_view

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def execute1(l_text, l_label, l_code, task, task_name,model_name= "beto", max_length = 128, opt="adam",learning_rate = 1e-5, batch_size = 8, epochs = 12, k=5, f=None, corpus=None):

  d_text_code={}
  for i in range(len(l_text)):
    d_text_code[l_text[i]]=  l_code[i]
  
  l_text=np.asarray(l_text)
  l_label=np.asarray(l_label)
  np.random.seed(23)
  torch.manual_seed(23)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state = 23)
  csv_train_path = 'train.csv'
  csv_dev_path = 'test.csv'
  l_devacc, l_devepoch, l_predictedT, l_labelT, l_textT = [], [], [], [],[]
  for i, (train_index, test_index) in enumerate(skf.split(l_text, l_label)):
    save_temporal_data(csv_train_path, csv_dev_path, l_text, l_label, train_index, test_index)
    model_base, tokenizer = get_model_and_tokenizer(model_name)
    model = Model(model_base,model_name=model_name)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer=getOptimizar(model, opt, learning_rate)
    trainloader = DataLoader(RawDataset(csv_train_path), batch_size=batch_size, shuffle=True, num_workers=4)
    devloader = DataLoader(RawDataset(csv_dev_path), batch_size=batch_size, shuffle=True, num_workers=4)

    max_devacc, max_epoch, l_predicAux, l_labelAux, l_textAux = 0,0,[],[],[]
    l_to_save_best = []
    for epoch in range(epochs):
      l_to_save = []
      running_loss = 0.0
      perc = 0
      acc = 0
      batches = len(trainloader)
      rloss=None

      #training
      model.output_and_layers = False
      for j, data in enumerate(trainloader, 0):
        torch.cuda.empty_cache()
        inputs, labels = data['text'], data['label'].to(device)
        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # outputs = model(**inputs)
        # print("ya")
        
        if model_name == "trans_xl":
          inputs = tokenizer(inputs, padding=True, return_tensors='pt').input_ids.to(device)
        else:
          inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)

        # inputs = tokenizer(inputs,  return_tensors='pt').input_ids.to(device)
        optimizer.zero_grad()
        # print(inputs.size())
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
          if j == 0:
            acc = ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy()
          else: acc = (acc + ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy())/2.0
          running_loss += loss.item()
        del inputs, labels, outputs

        if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
          perc = (1+j)*100.0/batches

          print('\r Epoch:{} step {} of {}. {}% loss: {}'.format(epoch+1, j+1, batches, np.round(perc, decimals=3), np.round(running_loss, decimals=3)), end="")
          rloss = running_loss
          running_loss = 0

      # Evaluating Acc of development
      with torch.no_grad():
        model.output_and_layers = True
        out = None
        log = None
        texts = None
        for k, data in enumerate(devloader, 0):
          torch.cuda.empty_cache()
          inputs, label = data['text'], data['label'].to(device)
          aux_texts = inputs
          # inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)

          # to save matrices
          inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)
          dev_out, layers = model(inputs)
          layers= [[layer[i].mean(dim=0,keepdim=True) for layer in layers] for i in range(inputs.shape[0])]
          # print(label)
          # print(dev_out)
          # print(torch.max(dev_out, 1))
          # print(torch.max(dev_out, 1).values)
          # print(torch.max(dev_out, 1).indices)
          # l_to_save.append([inputs, aux_texts, layers, label, torch.max(dev_out, 1).indices.squeeze().tolist(),torch.max(dev_out, 1).values.squeeze().tolist()])
          #
          # raise

          # model_view(attention, tokens)
          # print(attention)

          # save_attentions(attention,tokens)


          # dev_out = model(inputs)
          if k == 0:
            out = dev_out
            log = label
            texts = aux_texts
          else:
            out = torch.cat((out, dev_out), 0)
            log = torch.cat((log, label), 0)
            texts = texts+ aux_texts

        dev_loss = criterion(out, log).item()
        dev_acc = float(((torch.max(out, 1).indices == log).sum()/len(log)).cpu().numpy())


      # torch.save(model.state_dict(), 'epoch_{}_dev_loss{}.pt'.format(epoch+1, np.round(dev_loss, decimals=2)))
      print(" acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3), np.round(dev_loss, decimals=3), np.round(dev_acc, decimals=3)))
      if max_devacc<dev_acc:
        max_devacc=dev_acc
        max_epoch = epoch+1
        l_predicAux = out
        l_labelAux = log
        l_textAux = texts
        l_to_save_best = l_to_save[:]


      print("\n"+"max_devacc",max_devacc, "in", max_epoch)

    l_devacc.append(max_devacc)
    l_devepoch.append(max_epoch)
    if l_predictedT==[]:
      l_predictedT=l_predicAux
      l_labelT = l_labelAux
      l_textT = l_textAux
    else:
      l_predictedT=torch.cat((l_predictedT, l_predicAux), 0)
      l_labelT=torch.cat((l_labelT, l_labelAux), 0)
      l_textT = l_textT+ l_textAux
    for batch in l_to_save_best:
      save_matrices(batch[0], tokenizer, d_text_code,batch[1], batch[2], real_tag = batch[3], predicted_tag = batch[4], prob =  batch[5], i_or_ii="I" if task_name=="in_taxonomy" else "II")
    
    print('Training Finished Split: {}'.format(i))
    print()
    # print_conf_matrix(l_labelaux.squeeze().tolist(), torch.max(l_predicted, 1).indices.squeeze().tolist())
    if f != None:
      f.write("\n"+'Training Finished Split: {}'. format(i))
      f.write("\n"+'accuracies: {}'.format(l_devacc))
      f.write("\n"+'in epochs: {}'.format(l_devepoch))
    print("\n\n")

  print(l_devacc)
  import statistics
  average = np.round(statistics.mean(l_devacc), decimals=3)
  dest =  np.round(statistics.stdev(l_devacc), decimals=3)
  f.write("\n\n"+'acc: {} stdev: {}'.format(average, dest))

  ff = open("/content/drive/MyDrive/Stereotype/output/final", "a")
  ff.write("\n\n"+'acc: {} stdev: {}                      configuration {}'.format(average, dest, f.name))
  ff.close()
  
  

  # conf_examples(l_textT, l_labelT.squeeze().tolist(), torch.max(l_predictedT, 1).indices.squeeze().tolist(), d_text_code, corpus)
  print_conf_matrix(l_labelT.squeeze().tolist(), torch.max(l_predictedT, 1).indices.squeeze().tolist())


def execute2(task_name, f_load_source, f_load_target, model_name= "beto", max_length = 128, opt="adam",learning_rate = 1e-5, batch_size = 8, epochs = 12, k=5, f=None, corpus=None):


  source2target = f_load_source.split("_")[-1]+"2"+f_load_target.split("_")[-1]

  print(source2target)
  l_text_source,l_label_source, l_code_source = d_name_f[f_load_source]()
  l_text_target,l_label_target, l_code_target = d_name_f[f_load_target]()
  d_text_code={}
  for i in range(len(l_code_target)):
    d_text_code[l_text_target[i]] = l_code_target[i]

  l_text_training=np.asarray(l_text_source)#source/train
  l_label_training=np.asarray(l_label_source)#source/train
  l_text_dev=np.asarray(l_text_target)#source/dev
  l_label_dev=np.asarray(l_label_target)#target/dev

  np.random.seed(23)
  torch.manual_seed(23)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  csv_train_path = 'train.csv'
  csv_dev_path = 'test.csv'
  # iterar por varios split 20% para dev (u 80% para el dev y 20% para test)
  l_devacc, l_devepoch, l_predictedT, l_labelT, l_textT = [], [], [], [],[]
  save_temporal_data2(csv_train_path, csv_dev_path, l_text_training, l_label_training, l_text_dev, l_label_dev)
  model_base, tokenizer = get_model_and_tokenizer(model_name)
  model = Model(model_base,model_name=model_name)
  model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer=getOptimizar(model, opt, learning_rate)
  trainloader = DataLoader(RawDataset(csv_train_path), batch_size=batch_size, shuffle=True, num_workers=4)
  devloader = DataLoader(RawDataset(csv_dev_path), batch_size=batch_size, shuffle=True, num_workers=4)
  max_devacc, max_epoch, l_predicAux, l_labelAux, l_textAux = 0,0,[],[],[]
  l_to_save_best = []
  for epoch in range(epochs):
    l_to_save = []
    running_loss = 0.0
    perc = 0
    acc = 0
    batches = len(trainloader)
    rloss=None

    #training
    model.output_and_layers = False
    for j, data in enumerate(trainloader, 0):
      torch.cuda.empty_cache()
      inputs, labels = data['text'], data['label'].to(device)
      # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
      # outputs = model(**inputs)
      # print("ya")

      if model_name == "trans_xl":
        inputs = tokenizer(inputs, padding=True, return_tensors='pt').input_ids.to(device)
      else:
        inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)

      # inputs = tokenizer(inputs,  return_tensors='pt').input_ids.to(device)
      optimizer.zero_grad()
      # print(inputs.size())
      outputs, _ = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy()
        else: acc = (acc + ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy())/2.0
        running_loss += loss.item()
      del inputs, labels, outputs

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        perc = (1+j)*100.0/batches

        print('\r Epoch:{} step {} of {}. {}% loss: {}'.format(epoch+1, j+1, batches, np.round(perc, decimals=3), np.round(running_loss, decimals=3)), end="")
        rloss = running_loss
        running_loss = 0

    # Evaluating Acc of development
    with torch.no_grad():
      model.output_and_layers = True
      out = None
      log = None
      texts = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache()
        inputs, label = data['text'], data['label'].to(device)
        aux_texts = inputs
        # inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)

        # to save matrices
        inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(device)
        dev_out, layers = model(inputs)
        layers= [[layer[i].mean(dim=0,keepdim=True) for layer in layers] for i in range(inputs.shape[0])]
        # print(label)
        # print(dev_out)
        # print(torch.max(dev_out, 1))
        # print(torch.max(dev_out, 1).values)
        # print(torch.max(dev_out, 1).indices)
        l_to_save.append([inputs, aux_texts, layers, label, torch.max(dev_out, 1).indices.squeeze().tolist(),torch.max(dev_out, 1).values.squeeze().tolist()])
        #
        # raise

        # model_view(attention, tokens)
        # print(attention)

        # save_attentions(attention,tokens)


        # dev_out = model(inputs)
        if k == 0:
          out = dev_out
          log = label
          texts = aux_texts
        else:
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, label), 0)
          texts = texts+ aux_texts

      dev_loss = criterion(out, log).item()
      dev_acc = float(((torch.max(out, 1).indices == log).sum()/len(log)).cpu().numpy())


    # torch.save(model.state_dict(), 'epoch_{}_dev_loss{}.pt'.format(epoch+1, np.round(dev_loss, decimals=2)))
    print(" {} acc: {} ||| dev_loss: {} dev_acc: {}".format(source2target, np.round(acc, decimals=3), np.round(dev_loss, decimals=3), np.round(dev_acc, decimals=3)))
    if max_devacc<dev_acc:
      max_devacc=dev_acc
      max_epoch = epoch+1
      l_predicAux = out
      l_labelAux = log
      l_textAux = texts
      l_to_save_best = l_to_save[:]

    print("\n"+"max_devacc",max_devacc, "in", max_epoch)

  l_devacc.append(max_devacc)
  l_devepoch.append(max_epoch)
  if l_predictedT==[]:
    l_predictedT=l_predicAux
    l_labelT=l_labelAux
    l_textT = l_textAux
  else:
    l_predictedT=torch.cat((l_predictedT, l_predicAux), 0)
    l_labelT=torch.cat((l_labelT, l_labelAux), 0)
    l_textT = l_textT+ l_textAux
  print(len(l_to_save_best))
  for batch in l_to_save_best:
    save_matrices(inputs=batch[0], tokenizer = tokenizer, d_text_code=d_text_code,aux_texts = batch[1], layers = batch[2], real_tag = batch[3], predicted_tag = batch[4], prob =  batch[5], i_or_ii=source2target )

    print('Training Finished')
    print()
    # print_conf_matrix(l_labelaux.squeeze().tolist(), torch.max(l_predicted, 1).indices.squeeze().tolist())
    # if f != None:
    #   f.write("\n"+'Training Finished')
    #   f.write("\n"+'accuracies: {}'.format(l_devacc))
    #   f.write("\n"+'in epochs: {}'.format(l_devepoch))
    print("\n\n")


  # import statistics
  # average = np.round(statistics.mean(l_devacc), decimals=3)
  # dest =  np.round(statistics.stdev(l_devacc), decimals=3)
  # print(average, dest, source2target)
  if f != None:

    f.write("\n"+source2target)
    f.write("\n"+'accuracies: {}'.format(l_devacc))
    f.write("\n"+'in epochs: {}'.format(l_devepoch))
    f.write("\n\n"+'acc: {} in {} epoch'.format(max_devacc, max_epoch))

  ff = open("/home/jjsjunquera/Stereotype/output/deception/final_deception", "a")
  ff.write("\n\n"+'{}  acc: {} epoc: {}  configuration {} '.format(source2target, max_devacc, max_epoch, f.name))

  ff.close()



  # conf_examples(l_textT, l_labelT.squeeze().tolist(), torch.max(l_predictedT, 1).indices.squeeze().tolist(), d_text_code, corpus)
  print_conf_matrix(l_labelT.squeeze().tolist(), torch.max(l_predictedT, 1).indices.squeeze().tolist())


def execute(task_name,model_name= "bert-base-multilingual-cased", max_length = 128, opt="adam",learning_rate = 1e-5, batch_size = 64, epochs = 12, k=5, f=None):
  print("\n\n\n\n\n==============================================================================================================")
  print("\n\n\n\n\n==============================================================================================================")
  print("\n\n\n\n\n==============================================================================================================")
  print("\n\n\n\n\n==============================================================================================================")
  # epochs = 1
  task = TaskStereotype(task_name)
  from load_data import get_corpus
  corpus = get_corpus(task, statistics=True, use_lr=False)
  # corpus.d_code_maskresults[l_code[i]] = {"masked_text": x[i],
  #                                             "actual": y[i],
  #                                             "predicted": y_pred[i]}
  l_text, l_label, l_code = corpus.get_texts_and_labels(f_get_tax=corpus.get_tax1)
  # print(list(corpus.d_code_maskresults.items())[0])
  # raise
  
  # print("end")


  # l_text = apply_mask(l_text, l_mostFrequent = get_mostfrequent(corpus))

  execute1(l_text, l_label, l_code, task, task_name,model_name= model_name, max_length = max_length, opt=opt,learning_rate = learning_rate, batch_size = batch_size, epochs = epochs, k=k, f=f, corpus=corpus)





def getOptimizar(model,opt, lr):
  if opt=="rmsprop":
    return torch.optim.RMSprop(model.parameters(), lr=lr)
  else: return torch.optim.Adam(model.parameters(), lr=lr)


class RawDataset(Dataset):
  def __init__(self, csv_file):
    self.data_frame = pd.read_csv(csv_file)

  def __len__(self):
    return len(self.data_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    text  = self.data_frame.loc[idx, 'text']
    try:
      value = self.data_frame.loc[idx, 'label']
    except:
      value =  0.
    sample = {'text': text, 'label': value}
    return sample

def save_temporal_data(csv_train_path, csv_dev_path, l_text, l_label, train_index, test_index):
  data = l_text[train_index]
  label = l_label[train_index]
  dictionary = {'text': data, 'label':list(label)}
  df = pd.DataFrame(dictionary)
  df.to_csv(csv_train_path)
  data = l_text[test_index]
  label = l_label[test_index]
  dictionary = {'text': data, 'label':label}
  df = pd.DataFrame(dictionary)
  df.to_csv(csv_dev_path)

def save_temporal_data2(csv_train_path, csv_dev_path, l_text_t, l_label_t, l_text_d, l_label_d):
  data = l_text_t
  label = l_label_t
  dictionary = {'text': data, 'label':list(label)}
  df = pd.DataFrame(dictionary)
  df.to_csv(csv_train_path)
  data = l_text_d
  label = l_label_d
  dictionary = {'text': data, 'label':label}
  df = pd.DataFrame(dictionary)
  df.to_csv(csv_dev_path)

class Model(torch.nn.Module):
    def __init__(self, model_base, dr=0.3, model_name="trans_xl"):
        super(Model, self).__init__()
        self.encoder = model_base
        if model_name == "trans_xl":
          self.dense1 = torch.nn.Linear(in_features=1024, out_features=64, bias=True)
        else:
          self.dense1 = torch.nn.Linear(in_features=768, out_features=64, bias=True)


        self.drop= torch.nn.Dropout(p=dr)
        self.dense2 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)
        self.output_and_layers = False

    def forward(self, x):

        layers = None
        if self.output_and_layers:
          z = self.encoder(x)
          # print(len(z[-1]))
          # raise Exception()
          x =z.hidden_states[12][:,0]
          layers = z[-1]
        else:
          x = self.encoder(x).hidden_states[12][:,0]
        x = torch.nn.functional.relu(self.dense1(x))
        x=self.drop(x)
        x = torch.nn.functional.relu(self.dense2(x))
        x = self.classifier(x)
        x=self.softmax(x)

        return x, layers







