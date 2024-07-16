# Importing os, numpy and pandas for data manipulation
import os
import numpy as np 
import pandas as pd

# For data visualization, we will use matplotlib, wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# For data preprocessing, we will use Counter, train_test_split, Levenshtein distance, Python Image Library and OneHotEncoder
from collections import Counter
import Levenshtein as lev
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# For saving and loading the preprocessed data, we will use pickle
import pickle

# For Building the model, we will use PyTorch and its functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
from torch.utils.data import Dataset, DataLoader

# For taking the image from the URL, we will use requests
import requests

# For evaluation, we will need sklearn.metrics.average_precision_score
from sklearn.metrics import average_precision_score

# Importing json for results formatting which will be uploaded for evaluation
import json

# 2024/07/13
import sys
import tqdm
from torchvision import transforms
import argparse
import re
from statistics import mode
import csv

# ============================================================
# 
# ============================================================
def read_dataframe(path):
    """
    Reads the JSON file and returns a dataframe with the required columns (image, question, answers, answer_type, answerable)

    Parameters:
        path (str): Path to the JSON file

    Returns:
        df (pandas.DataFrame): Dataframe with the required columns
    """
    df = pd.read_json(path)
    df = df[['image', 'question', 'answers', 'answer_type', 'answerable']]
    return df

# ============================================================
# 
# ============================================================
def read_dataframe_DL24(path):
    """
    Reads the JSON file and returns a dataframe with the required columns (image, question, answers, answer_type, answerable)

    Parameters:
        path (str): Path to the JSON file

    Returns:
        df (pandas.DataFrame): Dataframe with the required columns
    """
    df = pd.read_json(path)
    #df = df[['image', 'question', 'answers', 'answer_type', 'answerable']]
    return df

# ============================================================
# 
# ============================================================
def split_train_test(dataframe, test_size = 0.05):
    """
    Splits the dataframe into train and test sets

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be split

    Returns:
        train (pandas.DataFrame): Train set
        test (pandas.DataFrame): Test set
    """
    train, test = train_test_split(dataframe, test_size=test_size, random_state=42, stratify=dataframe[['answer_type', 'answerable']])
    return train, test

# ============================================================
# 
# ============================================================
def plot_histogram(dataframe, column):
    """
    Plots the histogram of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted
    
    Returns:
        None
    """
    plt.hist(dataframe[column])
    plt.title(column)
    plt.show()

# ============================================================
# 
# ============================================================
def plot_pie(dataframe, column):
    """
    Plots the pie chart of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted
    
    Returns:
        None
    """
    value_counts = dataframe[column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    plt.title(column)
    plt.show()

# ============================================================
# 
# ============================================================
def plot_wordcloud(dataframe, column):
    """
    Plots the wordcloud of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted

    Returns:
        None
    """
    text = " ".join([word for word in dataframe[column]])

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    min_font_size = 10).generate(text) 
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

# ============================================================
# 
# ============================================================
def explore_dataframe(dataframe):
    """
    Explores the dataframe (EDA) by plotting the pie charts, histograms and wordclouds of the columns

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be explored

    Returns:
        None
    """
    plot_pie(dataframe, 'answer_type')
    plot_pie(dataframe, 'answerable')
    plot_histogram(dataframe, 'answerable')
    plot_wordcloud(dataframe, 'question')
    
# ============================================================
# 
# ============================================================
def get_number_of_distinct_answers(dataframe):
    """
    Returns the number of distinct answers in the dataframe

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be explored

    Returns:
        len(unique_answers_set) (int): Number of distinct answers in the dataframe
    """
    unique_answers_set = set()
    for row in dataframe['answers']:
        for answer_map in row:
            unique_answers_set.add(answer_map['answer'])
    return len(unique_answers_set)

# ============================================================
# 
# ============================================================
def process_images(dataframe, image_path, clip_model, preprocessor, device):
    """
    Processes the images in the dataframe and returns the image features

    Parameters:
        dataframe (pandas.DataFrame): Dataframe containing the images
        image_path (str): Path to the input images
        clip_model (clip.model.CLIP): CLIP model
        preprocessor (clip.model.Preprocess): Preprocessor for the CLIP model
        device (torch.device): Device to be used for processing
    
    Returns:
        images (list): List of image features
    """
    images = []
    idx=0
    #for _, row in dataframe.iterrows():
    for _, row in tqdm.tqdm(dataframe.iterrows()):
        full_path = image_path + "/" + row['image']
        image = Image.open(full_path)
        image = preprocessor(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        image_features = torch.flatten(image_features, start_dim=1)
        images.append(image_features)
        idx+=1
        #if idx>2:
        #    break
    return images

# ============================================================
# 
# ============================================================
def process_questions(dataframe, clip_model,device):
    """
    Processes the questions in the dataframe and returns the question features

    Parameters:
        dataframe (pandas.DataFrame): Dataframe containing the questions
        clip_model (clip.model.CLIP): CLIP model
        device (torch.device): Device to be used for processing

    Returns:
        questions (list): List of question features
    """
    questions = []
    idx=0
    #for _, row in dataframe.iterrows():
    for _, row in  tqdm.tqdm(dataframe.iterrows()):
        question = row['question']
        question =  clip.tokenize(question).to(device)
        text_features = clip_model.encode_text(question).float()
        text_features = torch.flatten(text_features, start_dim=1)
        questions.append(text_features)
        idx+=1
        #if idx>2:
        #    break
    return questions

# ============================================================
# 
# ============================================================
def Create_DafaFrame(ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH):

    train_df = read_dataframe(ANNOTATIONS_TRAIN_PATH)
    validation_df = read_dataframe(ANNOTATIONS_VAL_PATH)
    train_df, test_df = split_train_test(train_df, test_size=0.05)
    ANSWER_SPACE = get_number_of_distinct_answers(train_df) # The answer space will be decreased later when we process the answers
    print("Number of distinct answers: ", ANSWER_SPACE)

    df = [train_df, validation_df, test_df]

    return df, ANSWER_SPACE

# ============================================================
# 
# ============================================================
def Create_DafaFrame_2(ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH):

    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)

    train_df = read_dataframe(ANNOTATIONS_TRAIN_PATH)
    validation_df = read_dataframe(ANNOTATIONS_VAL_PATH)
    #train_df, test_df = split_train_test(train_df, test_size=0.05)
    #test_df = read_dataframe(ANNOTATIONS_TRAIN_PATH)
    ANSWER_SPACE = get_number_of_distinct_answers(train_df) # The answer space will be decreased later when we process the answers
    print("\t{} : (train_df) Number of distinct answers: ", DBG_TAG, ANSWER_SPACE)

    df = [train_df, validation_df]

    return df, ANSWER_SPACE

# ============================================================
# 
# ============================================================
def Create_DafaFrame_DL24(ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH):

    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)

    train_df      = read_dataframe_DL24(ANNOTATIONS_TRAIN_PATH)
    validation_df = read_dataframe_DL24(ANNOTATIONS_VAL_PATH)
    #train_df, test_df = split_train_test(train_df, test_size=0.05)
    #test_df       = read_dataframe_DL24(ANNOTATIONS_TRAIN_PATH)
    ANSWER_SPACE = get_number_of_distinct_answers(train_df) # The answer space will be decreased later when we process the answers
    print("\t{} : (train_df) Number of distinct answers: ", DBG_TAG, ANSWER_SPACE)

    #df = [train_df, validation_df, test_df]
    df = [train_df, validation_df]

    return df, ANSWER_SPACE

# ============================================================
# 
# ============================================================
def explore_3_dataframe(train_df, validation_df, test_df, flg_explore_df=False):

    if flg_explore_df:
       explore_dataframe(train_df)
    print("(train_df) Number of distinct answers: ", get_number_of_distinct_answers(train_df))
    print("(train_df) Number of samples in train: ", len(train_df))

    if flg_explore_df:
        explore_dataframe(validation_df)
    print("(validation_df) Number of distinct answers: ", get_number_of_distinct_answers(validation_df))
    print("(validation_df) Number of samples in validation set: ", len(validation_df))

    if flg_explore_df:
        explore_dataframe(test_df)
    print("(test_df) Number of distinct answers: ", get_number_of_distinct_answers(test_df))
    print("(test_df) Number of samples in test: ", len(test_df))

    return

# ============================================================
# 
# ============================================================
def Pickle_DF(train_df, validation_df, test_df, clip_model, preprocessor, TRAIN_PATH, VALIDATION_PATH, OUTPUT_PATH, DEVICE="cpu"):

    print("\t{} : Start process_images & process_questions".format(DBG_TAG))

    training_images = process_images(train_df, TRAIN_PATH, clip_model, preprocessor, DEVICE)
    print("\t{} : process_images of training Done".format(DBG_TAG))
    training_questions = process_questions(train_df, clip_model, DEVICE)
    print("\t{} : process_questions of training Done".format(DBG_TAG))
    with open(OUTPUT_PATH + 'training_images.pkl', 'wb') as f:
        pickle.dump(training_images, f)
    with open(OUTPUT_PATH + 'training_questions.pkl', 'wb') as f:
        pickle.dump(training_questions, f)

    validation_images = process_images(validation_df, VALIDATION_PATH, clip_model, preprocessor, DEVICE)
    validation_questions = process_questions(validation_df, clip_model, DEVICE)
    with open(OUTPUT_PATH + 'validation_images.pkl', 'wb') as f:
        pickle.dump(validation_images, f)
    with open(OUTPUT_PATH + 'validation_questions.pkl', 'wb') as f:
        pickle.dump(validation_questions, f)

    test_images = process_images(test_df, TRAIN_PATH, clip_model, preprocessor, DEVICE)
    test_questions = process_questions(test_df, clip_model, DEVICE)
    with open(OUTPUT_PATH + 'test_images.pkl', 'wb') as f:
        pickle.dump(test_images, f)
    with open(OUTPUT_PATH + 'test_questions.pkl', 'wb') as f:
        pickle.dump(test_questions, f)

    return

# ============================================================
# 
# ============================================================
def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================================
# 
# ============================================================
class VizWizDataset(Dataset):
#    def __init__(self, dataframe, answer_type_onehotencoder = None, answer_onehotencoder = None, model_name = "RN50x64", 
#                 images_features = torch.tensor([]), questions_features = torch.tensor([]), image_path="",
#                 dl24_json=False, flg_Pickle=True, q_onehot_id=0,
#                 clip_model=None, preprocessor=None, have_answer=True, device="cpu"):
    def __init__(self, dfs, answer_type_onehotencoder = None, answer_onehotencoder = None, model_name = "RN50x64", 
                 image_path="", add_answer_path=None,
                 dl24_json=False, flg_Pickle=True, q_onehot_id=0,
                 clip_model=None, preprocessor=None, have_answer=True, device="cpu"):
        super(VizWizDataset, self).__init__()

        # 2024/07/12
        self.dl24_json = dl24_json
        self.have_answer = have_answer
        self.q_onehot_id = q_onehot_id
        # question / answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}
        self.answer2idx_csv = {}
        self.idx2answer_csv = {}

        # Total counter for all answers before filtering, used in Tie Breaking when building the answer vocabulary
        if self.have_answer:
            self.answer_counter = Counter() 

        # Saving the dataframe
        self.dataframe = dfs[0]

        # Saving image & question embeddings
        #self.images_features = images_features
        #self.questions_features = questions_features
        if self.have_answer and not self.dl24_json:
            self.answerable = self.dataframe['answerable'].to_numpy()
        
        # List for answers for each question (each question has 10 answers)
        if self.have_answer:
            self.answer_counter_per_question = []

        # Populating the counter for words in answers which will be used when building answer vocabulary
        if self.have_answer:
            self.build_answer_counter()

        # Building the answer vocabulary according to the methodology explained in the paper
        if self.have_answer:
            self.build_answer_vocab()

        # The number of vocabulary words after filtering
        if self.have_answer:
            print("Number of distinct answers: ", len(self.get_answer_vocab()))

        # One hot encoding the answers
        if self.have_answer and not self.dl24_json:
            if answer_type_onehotencoder is None:
                answer_type_onehotencoder = OneHotEncoder(handle_unknown='ignore')
                answer_type_onehotencoder.fit(self.copied_dataframe[['answer_type']])
        
        # One hot encoding the answer types
        if self.have_answer:
            if answer_onehotencoder is None:
                answer_onehotencoder = OneHotEncoder(handle_unknown='ignore')
                answer_onehotencoder.fit(self.copied_dataframe[['answer']])
        
        # Saving the one hot encoders
        if self.have_answer:
            self.answer_onehotencoder = answer_onehotencoder
        if self.have_answer and not self.dl24_json:
            self.answer_type_onehotencoder = answer_type_onehotencoder

        # Transforming the answers and answer types to one hot encoded vectors
        if self.have_answer:
            self.answer_onehotencoded = answer_onehotencoder.transform(self.copied_dataframe[['answer']]).toarray()
        if self.have_answer and not self.dl24_json:
            self.answer_type_onehotencoded = answer_type_onehotencoder.transform(self.copied_dataframe[['answer_type']]).toarray()
        
        # Saving the answer categories (vocabulary) which will be used when getting index of the predicted answer
        if self.have_answer:
            self.answers_categories = self.answer_onehotencoder.categories_[0].tolist()
            #print("self.answers_categories[0]={}".format(self.answers_categories[0]))

        # Saving answers for each question (each question has 10 answers)
        if self.have_answer:
            self.build_answer_counter_per_question()
        
        # 回答に含まれる単語を辞書に追加
        if self.have_answer:
            for df in dfs:
                self.make_answer2idx(df["answers"])
            #self.make_answer2idx(self.dataframe["answers"])
            #if df_list is not None:
            #    for df in df_list:
            #        self.make_answer2idx(df["answers"])
            if add_answer_path is not None:
                self.add_answer2idx_by_csv(add_answer_path)

        # 2024/07/13
        self.image_path = image_path
        self.flg_Pickle = flg_Pickle
        list_da_pp = []
        list_da_pp.append(transforms.ToTensor())
        self.transform = transforms.Compose(list_da_pp)   # DA & PP
        self.clip_model = clip_model
        self.preprocessor = preprocessor
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        #print("{}, {}, {}".format(index, type(self.dataframe['image']), self.dataframe['image'][index]))
        image_full_path = "{}/{}".format(self.image_path, self.dataframe['image'][index])
        image = Image.open(image_full_path)
        #image = self.transform(image)
        #image = self.transform(image)
        image = self.preprocessor(image).unsqueeze(0).to(self.device)
        image_features2 = self.clip_model.encode_image(image)

        q_txt = self.dataframe['question'][index]
        question =  clip.tokenize(q_txt).to(self.device)
        text_features = clip_model.encode_text(question).float()
        text_features = torch.flatten(text_features, start_dim=1)
        #print("{}, {}, {}".format(index, image_full_path, q_txt))

        if self.have_answer:
            answer = torch.tensor(self.answer_onehotencoded[index], dtype=torch.float32)
        if self.have_answer and not self.dl24_json:
            answer_type = torch.tensor(self.answer_type_onehotencoded[index], dtype=torch.float32)
        if self.have_answer:
            answer_counter = torch.tensor(self.answer_counter_per_question[index], dtype=torch.long)
        if self.have_answer and not self.dl24_json:
            answerable = torch.tensor(self.answerable[index], dtype=torch.float32)

        # base_line
        if self.have_answer:
            answers_id = [self.answer2idx[process_text(answer["answer"])] for answer in self.dataframe["answers"][index]]
            answers_id = torch.tensor(answers_id)
            mode_answer_idx = mode(answers_id)  # 最頻値を取得（正解ラベル）
            mode_answer_idx = int(mode_answer_idx)

        if self.have_answer:

            #if self.flg_Pickle and not self.dl24_json:
            #    return self.images_features[index], self.questions_features[index], answer, answer_type, answer_counter, answerable, answers_id, mode_answer_idx
            #else:
            if not self.dl24_json:
                return image_features2, text_features, answer, answer_type, answer_counter, answerable, answers_id, mode_answer_idx
            else:
                return image_features2, text_features, answer, answer_counter, answers_id, mode_answer_idx
        else:
            #if self.flg_Pickle and not self.dl24_json:
            #    return self.images_features[index], self.questions_features[index], answer, answer_type, answer_counter, answerable
            #else:
            if not self.dl24_json:
                return image_features2, text_features
            else:
                return image_features2, text_features

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def build_answer_counter_per_question(self):
        if self.have_answer:
            for index, row in self.dataframe.iterrows():
                temp_list = []
                for answer_map in row['answers']:
                    answer = answer_map['answer']
                    # check if answer in self.answers_categories
                    if answer in self.answers_categories:
                        answer_index = self.answers_categories.index(answer)
                        temp_list.append(answer_index)
                # Torch.tensor requires the all the lists to have constant length, so we pad the list with -1 if needed
                while len(temp_list) < 10:
                    temp_list.append(-1)
                self.answer_counter_per_question.append(temp_list)
        else:
            pass
    
    def build_answer_vocab(self):
        if self.have_answer:
            # Building answer vocab follow this policy:
            # for each question we have 10 answers, we choose the most frequent answer as the answer for this question
            # if there is a tie, we choose the most common one in the whole dataset
            # if there is a tie, we choose the pairwise Levenshtein distance is used to find the answer that is most representative to all others.
            
            # Copying the original dataframe which will be manipulated
            self.copied_dataframe = self.dataframe.copy()
            self.copied_dataframe.drop(columns=['answers'], inplace=True)

            # Adding extra column named 'answer'
            self.copied_dataframe['answer'] = None

            for index, row in self.dataframe.iterrows():
                intermediate_counter = Counter()
                for answer_map in row['answers']:
                    answer = answer_map['answer']
                    intermediate_counter.update([answer])
                
                # let's see the top elements in the answers_counter to check if there is a tie
                top_elements = intermediate_counter.most_common(1)
                if len(top_elements) == 1:
                    self.copied_dataframe.at[index, 'answer'] = top_elements[0][0]
                else:
                    # let's see who is the most common answer in the whole dataset
                    top_elements = self.answer_counter.most_common(1)
                    if len(top_elements) == 1:
                        self.copied_dataframe.at[index, 'answer'] = top_elements[0][0]
                    else:
                        # let's get the minimum levenshtein distance between the answers in top_elements
                        current_min = np.inf
                        current_answer = None
                        for answer in top_elements:
                            total_distance = 0
                            for answer2 in top_elements:
                                if answer != answer2:
                                    lev_distance = lev.distance(answer[0], answer2[0])
                                    total_distance += lev_distance
                            if total_distance < current_min:
                                current_min = total_distance
                                current_answer = answer[0]
                        self.copied_dataframe.at[index, 'answer'] = current_answer
        else:
            pass
        return

    def build_answer_counter(self):
        if self.have_answer:
            for row in self.dataframe['answers']:
                for answer_map in row:
                    self.answer_counter.update([answer_map['answer']])
        else:
            pass
    
    def get_answer_vocab(self):
        if self.have_answer:
            return self.copied_dataframe['answer'].unique()
        else:
            return 0
        
    def make_answer2idx(self, df_answers):
        # ----------------------------------------
        # answer 辞書
        # ----------------------------------------
        if self.have_answer:
            # 回答に含まれる単語を辞書に追加
            #for answers in self.df["answers"]: # answers = [ { "answer_confidence": "yes", "answer": "spray bottle" }, { "answer_confidence": "yes", "answer": "multipurpose spray bottle" }, .. ]
            for answers in df_answers: # answers = [ { "answer_confidence": "yes", "answer": "spray bottle" }, { "answer_confidence": "yes", "answer": "multipurpose spray bottle" }, .. ]
                #print("answers={}".format(answers))
                for answer in answers: # answer = { "answer_confidence": "yes", "answer": "spray bottle" }
                    #print("answer={}".format(answer))
                    word = answer["answer"] # "spray bottle"
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
            print("\t{} : len(self.answer2idx)-1={}, len(self.idx2answer)-1={}".format(DBG_TAG, len(self.answer2idx), len(self.idx2answer)))

        return

    def add_answer2idx_by_csv(self, add_answer_csv):
        # ----------------------------------------
        # answer 辞書
        # ----------------------------------------
        with open(add_answer_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                #print(row)
                word = row[0]
                word = process_text(word)
                if word not in self.answer2idx_csv:
                    #print("New word in answer = {}".format(word))
                    self.answer2idx_csv[word] = len(self.answer2idx_csv)
                if word not in self.answer2idx:
                    #print("New word in answer = {}".format(word))
                    self.answer2idx[word] = len(self.answer2idx)

        self.idx2answer_csv = {v: k for k, v in self.answer2idx_csv.items()}  # 逆変換用の辞書(answer)
        self.idx2answer     = {v: k for k, v in self.answer2idx.items()}      # 逆変換用の辞書(answer)
        print("\t{} : len(self.answer2idx_csv)={}, len(self.idx2answer_csv)={}".format(DBG_TAG, len(self.answer2idx_csv), len(self.idx2answer_csv)))
        print("\t{} : len(self.answer2idx)-2={}, len(self.idx2answer)-2={}".format(DBG_TAG, len(self.answer2idx), len(self.idx2answer)))


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)): # 10人
            num_match = 0
            for j in range(len(answers)): # 10人
                if i == j:
                    continue
                if pred == answers[j]: # 10人から順に1人を除外、残り9人のうち回答が最頻値と合っていればカウント
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# ============================================================
# 
# ============================================================
class VQAModel(nn.Module):

    def __init__(self, num_classes, hidden_size, model_name = "ViT-L/14@336px", device = torch.device("cpu")):
        super(VQAModel, self).__init__()

        self.training_losses = []
        self.validation_losses = []

        self.training_accuracies = []
        self.validation_accuracies = []

        self.vizwiz_training_accuracies = []
        self.vizwiz_validation_accuracies = []

        self.training_answerability = []
        self.validation_answerability = []
        
        self.device = device
        self.model_name = model_name

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Initializing Binary Cross Entropy Loss which will be used to train the model on answerability
        self.answerability_loss_fn = nn.BCELoss()
        
        # Loading the CLIP model
        self.clip_model, self.preprocess = clip.load(model_name, device = device)
        
        # Freezing the CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # First linear layer
        self.linear_layer1 = nn.Sequential(
            nn.LayerNorm(self.clip_model.visual.output_dim + self.clip_model.text_projection.shape[1]),
            nn.Dropout(p=0.5),
            nn.Linear(self.clip_model.visual.output_dim + self.clip_model.text_projection.shape[1], hidden_size)
        )

        # Second linear layer
        self.linear_layer2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes) 
        )

        self.answer_type_layer = nn.Linear(hidden_size, 4)
        self.answer_mask_layer = nn.Linear(4, num_classes)

        self.sigmoid = nn.Sigmoid()

        # Answerability Linear Layer (We removed drop out layer because training answerability was very bad)
        self.answerability_linear_layer = nn.Sequential(
            nn.LayerNorm(self.clip_model.visual.output_dim + self.clip_model.text_projection.shape[1]),
            nn.Linear(self.clip_model.visual.output_dim + self.clip_model.text_projection.shape[1], hidden_size)
        )

        # Answerability Sigmoid Layer
        self.answerability_final_layer = nn.Linear(hidden_size, 1)

        # Sigmoid Layer for Answerability
        self.answerability_sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------
    def forward(self, image, question, dl24_json=False):

        # Flattening and concatenating the image and question features
        image = torch.flatten(image, start_dim=1)
        question = torch.flatten(question, start_dim=1)
        features = torch.cat((image, question), dim=1)
        
        # Calculating the answerability score
        if not dl24_json:
            answerability_score = self.answerability_linear_layer(features)
            answerability_score = self.answerability_final_layer(answerability_score)
            answerability_score = self.answerability_sigmoid(answerability_score)
            answerability_score = answerability_score.squeeze()
        
        # Passing the features through the first linear layer
        features = self.linear_layer1(features)

        if not dl24_json:
            # Passing the features to get 4 answer types
            answer_type = self.answer_type_layer(features)

            # Expanding answer make to the same size as the number of classes (vocab size)
            answer_mask = self.answer_mask_layer(answer_type)

            # Applying sigmoid to get the answer mask
            answer_mask = self.sigmoid(answer_mask)

        # Passing the features through the second linear layer
        output = self.linear_layer2(features)

        # Applying the answer mask to the output
        if not dl24_json:
            output = output * answer_mask
        
        if not dl24_json:
            return output, answer_type, answerability_score
        else:
            return output
    
    # ------------------------------------------------------------
    def train_model(self, training_dataloader, validation_dataloader, test_dataloader, 
                    criterion, optimizer, epochs = 10, dl24_json=False, filehead="", save_path = None, save_every = 1):

        for epoch in range(1,epochs+1):

            out_tr_step = self.training_step(training_dataloader, criterion, optimizer, dl24_json=dl24_json, device=self.device)

            if not dl24_json:
                #training_loss, training_accuracy, training_vizwiz_accuracy, train_answerability_score = out_tr_step
                training_loss, training_accuracy, training_vizwiz_accuracy = out_tr_step
            else:
                training_loss, training_accuracy, training_vizwiz_accuracy, = out_tr_step

            #validation_loss, validation_accuracy, validation_vizwiz_accuracy, validation_answerability_score = \
            #        self.validation_step(validation_dataloader, criterion, self.device)

            #test_accuracy, test_vizwiz_accuracy, test_answerability_score = \
            #        self.test_step(test_dataloader)

            self.training_losses.append(training_loss)
            #self.validation_losses.append(validation_loss)

            self.training_accuracies.append(training_accuracy)
            #self.validation_accuracies.append(validation_accuracy)

            self.vizwiz_training_accuracies.append(training_vizwiz_accuracy)
            #self.vizwiz_validation_accuracies.append(validation_vizwiz_accuracy)

            #if not dl24_json:
                #self.training_answerability.append(train_answerability_score)
                #self.validation_answerability.append(validation_answerability_score)
                        
            #print("Epoch: {} | Training Loss: {:.3f} | Validation Loss: {:.3f}".format(epoch, training_loss, validation_loss))
            #print("Epoch: {} | Training Accuracy: {:.3f} | Validation Accuracy: {:.3f} | Test Accuracy: {:.3f}".format(epoch, training_accuracy, validation_accuracy, test_accuracy))
            #print("Epoch: {} | Training VizWiz Accuracy: {:.3f} | Validation VizWiz Accuracy: {:.3f} | Test VizWiz Accuracy: {:.3f}".format(epoch, training_vizwiz_accuracy, validation_vizwiz_accuracy, test_vizwiz_accuracy))
            #print("Epoch: {} | Training Answerability Score: {:.3f} | Validation Answerability Score: {:.3f} | Test Answerability Score: {:.3f}\n".format(epoch, train_answerability_score, validation_answerability_score, test_answerability_score))
            
            if save_path != None and epoch % save_every == 0:
                self.save_model(save_path + "model_{}_e{}.pth".format(filehead, epoch))
        return
    
    # ------------------------------------------------------------
    def eval_model(self, validation_dataloader, 
                    criterion, epochs = 10, dl24_json=False):

        for epoch in range(1,epochs+1):

            #out_tr_step = self.training_step(training_dataloader, criterion, optimizer, dl24_json=dl24_json, device=self.device)

            #if not dl24_json:
            #    training_loss, training_accuracy, training_vizwiz_accuracy, train_answerability_score = out_tr_step
            #else:
            #    training_loss, training_accuracy, training_vizwiz_accuracy, = out_tr_step

            validation_loss, validation_accuracy, validation_vizwiz_accuracy, validation_answerability_score = \
                self.validation_step(validation_dataloader, criterion, dl24_json=dl24_json, device=self.device)

            #test_accuracy, test_vizwiz_accuracy, test_answerability_score = \
            #        self.test_step(test_dataloader)

            #self.training_losses.append(training_loss)
            self.validation_losses.append(validation_loss)

            #self.training_accuracies.append(training_accuracy)
            self.validation_accuracies.append(validation_accuracy)

            #self.vizwiz_training_accuracies.append(training_vizwiz_accuracy)
            self.vizwiz_validation_accuracies.append(validation_vizwiz_accuracy)

            if not dl24_json:
                #self.training_answerability.append(train_answerability_score)
                self.validation_answerability.append(validation_answerability_score)
                        
            #print("Epoch: {} | Training Loss: {:.3f} | Validation Loss: {:.3f}".format(epoch, training_loss, validation_loss))
            #print("Epoch: {} | Training Accuracy: {:.3f} | Validation Accuracy: {:.3f} | Test Accuracy: {:.3f}".format(epoch, training_accuracy, validation_accuracy, test_accuracy))
            #print("Epoch: {} | Training VizWiz Accuracy: {:.3f} | Validation VizWiz Accuracy: {:.3f} | Test VizWiz Accuracy: {:.3f}".format(epoch, training_vizwiz_accuracy, validation_vizwiz_accuracy, test_vizwiz_accuracy))
            #print("Epoch: {} | Training Answerability Score: {:.3f} | Validation Answerability Score: {:.3f} | Test Answerability Score: {:.3f}\n".format(epoch, train_answerability_score, validation_answerability_score, test_answerability_score))

        return
    
    # ------------------------------------------------------------
    def training_step(self, dataloader, criterion, optimizer, dl24_json=False, device="cpu"):

        DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)

        print("\t{} : Start".format(DBG_TAG))
        #print("\t{} : len(dataloader)={}".format(DBG_TAG, len(dataloader)))
        training_loss, training_accuracy, vizwiz_accuracy, total_sum = 0.0, 0.0, 0.0, 0
        loss_sum = 0
        answerable_true = []
        answerable_predicted = []
        self.train()
        ans_type_id = True
        #for _, batch in enumerate(dataloader):
        idx=0
        for batch in tqdm.tqdm(dataloader):
            #if idx>1:
            #    break

            if not dl24_json:                
                image, question, answer, answer_type, answers_for_questions, answerable, answers_id, mode_answer_idx = batch
                image, question, answer, answer_type, answers_for_questions, answerable, answers_id, mode_answer_idx = \
                    image.to(device), question.to(device), answer.to(device), answer_type.to(device), answers_for_questions.to(device), answerable.to(device), answers_id.to(device), mode_answer_idx.to(device)
            else:
                image, question, answer, answers_for_questions, answers_id, mode_answer_idx = batch
                image, question, answer, answers_for_questions, answers_id, mode_answer_idx = \
                    image.to(device), question.to(device), answer.to(device), answers_for_questions.to(device), answers_id.to(device), mode_answer_idx.to(device)
            #print("{} : image.size={}".format(i, np.shape(image)))
            #print("{} : question={}".format(i, question))

            #image, question, answer, answer_type, answers_for_questions, answerable = image.to(device), question.to(device), answer.to(device), answer_type.to(device), answers_for_questions.to(device), answerable.to(device)
            optimizer.zero_grad()
            if not dl24_json:
                output, answer_type_predicted, answerable_predict = self.forward(image, question)
            else:
                output = self.forward(image, question, dl24_json=dl24_json)
            if not dl24_json:
                answerable = 1 - answerable
                answerable_predict = 1.0 - answerable_predict

            if idx==0: 
                #print("\t{} : type(image)={}".format(DBG_TAG, type(image)))
                #print("\t{} : type(question)={}".format(DBG_TAG, type(question)))
                #print("\t{} : type(output)={}".format(DBG_TAG, type(output)))
                print("\t{} : output.size()={}".format(DBG_TAG, output.size()))
                print("\t{} : answer.size()={}".format(DBG_TAG, answer.size()))
                print("\t{} : answers_id.size()={}".format(DBG_TAG, answers_id.size()))
                print("\t{} : mode_answer_idx.squeeze()={}".format(DBG_TAG, mode_answer_idx.squeeze()))

            if not dl24_json:
                if ans_type_id:
                    loss = criterion(output, mode_answer_idx.squeeze()) \
                        + criterion(answer_type_predicted, answer_type) + self.answerability_loss_fn(answerable_predict, answerable)
                else:
                    loss = criterion(output, answer) \
                        + criterion(answer_type_predicted, answer_type) + self.answerability_loss_fn(answerable_predict, answerable)
            else:
                if ans_type_id:
                    loss = criterion(output, mode_answer_idx.squeeze())
                else:
                    loss = criterion(output, answer)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            loss_sum += 1

            # New
            if ans_type_id:

                predicted_answer1 = output.argmax(1)
                predicted_answer2 = torch.argmax(output, dim=1)
                #print("\t{} : output.argmax(1)={}".format(DBG_TAG, output.argmax(1)))
                #print("\t{} : torch.argmax(output, dim=1)={}".format(DBG_TAG, torch.argmax(output, dim=1)))
                #print("\t{} : answers_id.size()={}".format(DBG_TAG, answers_id.size()))

                actual_answer  = mode_answer_idx
                VQA_tmp = VQA_criterion(predicted_answer1, answers_id)
                #print("\t{} : VQA_tmp={}".format(DBG_TAG, VQA_tmp))
                vizwiz_accuracy += VQA_tmp
                total_sum +=1

            # Old
            else:

                predicted_answer = torch.argmax(output, dim = 1)
                #print("\t{} : predicted_answer={}".format(DBG_TAG, predicted_answer))
                actual_answer = torch.argmax(answer, dim = 1)
                for i in range(len(answer)):
                    if actual_answer[i] == predicted_answer[i]:
                        training_accuracy +=1
                    total_sum +=1
                    vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                    if not dl24_json:
                        answerable_true.append(answerable[i].item())
                        answerable_predicted.append(answerable_predict[i].item())


            print("\t{} : tr_loss={}, vizwiz_acc={}".format(DBG_TAG, training_loss/loss_sum, vizwiz_accuracy/total_sum))
            idx+=1

        #if not dl24_json:
        #    answerable_true = np.array(answerable_true)
        #    answerable_predicted = np.array(answerable_predicted)

        training_loss /= len(dataloader)
        training_accuracy /= total_sum
        vizwiz_accuracy /= total_sum
        
        if not dl24_json:
            #return training_loss, training_accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')
            return training_loss, training_accuracy, vizwiz_accuracy
        else:
            return training_loss, training_accuracy, vizwiz_accuracy

    def validation_step(self, dataloader, criterion, dl24_json=False, device="cpu"):

        DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)
        print("\t{} : Start".format(DBG_TAG))

        validation_loss, validation_accuracy, vizwiz_accuracy, total_sum = 0.0, 0.0, 0.0, 0
        answerable_true = []
        answerable_predicted = []
        ans_type_id = True

        self.eval()
        with torch.no_grad():
            #for _, batch in enumerate(dataloader):
            for batch in tqdm.tqdm(dataloader):
                image, question, answer, answer_type, answers_for_questions, answerable, answers_id, mode_answer_idx = \
                    batch
                image, question, answer, answer_type, answers_for_questions, answerable, answers_id, mode_answer_idx = \
                    image.to(device), question.to(device), answer.to(device), answer_type.to(device), answers_for_questions.to(device), answerable.to(device), answers_id.to(device), mode_answer_idx.to(device)

                if not dl24_json:
                    output, answer_type_predicted, answerable_predict = self.forward(image, question)
                else:
                    output = self.forward(image, question)
                
                # Answerablity is the confidence that quesion is not answerable, so we have to subtract from 1
                answerable = 1 - answerable
                answerable_predict = 1.0 - answerable_predict

                # New
                if ans_type_id:
                    loss = criterion(output, mode_answer_idx.squeeze()) + \
                        criterion(answer_type_predicted, answer_type) + \
                        self.answerability_loss_fn(answerable_predict, answerable)
                else:
                    loss = criterion(output, answer) + \
                        criterion(answer_type_predicted, answer_type) + \
                        self.answerability_loss_fn(answerable_predict, answerable)

                validation_loss += loss.item()
                predicted_answer = torch.argmax(output, dim = 1)

                # New
                if ans_type_id:

                    actual_answer = torch.argmax(answer, dim = 1)
                    for i in range(len(answer)):
                        if torch.sum(answer[i]) == 0:
                            continue
                        if actual_answer[i] == predicted_answer[i]:
                            validation_accuracy += 1
                        total_sum +=1
                        vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                        answerable_true.append(answerable[i].item())
                        answerable_predicted.append(answerable_predict[i].item())
                # Old
                else:

                    actual_answer = torch.argmax(answer, dim = 1)
                    for i in range(len(answer)):
                        if torch.sum(answer[i]) == 0:
                            continue
                        if actual_answer[i] == predicted_answer[i]:
                            validation_accuracy += 1
                        total_sum +=1
                        vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                        answerable_true.append(answerable[i].item())
                        answerable_predicted.append(answerable_predict[i].item())

                    
        answerable_true = np.array(answerable_true)
        answerable_predicted = np.array(answerable_predicted)
        
        validation_loss /= len(dataloader)
        validation_accuracy /= total_sum
        vizwiz_accuracy /= total_sum
        
        # We will use weighted average since that there is imbalance in answerability in the dataset as displayed in EDA section
        return validation_loss, validation_accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')
    
    def save_submission(self, dataloader, dataset, criterion, dl24_json=False, filehead="", npy_dir="npy", device="cpu"):

        DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)
        print("\t{} : Start".format(DBG_TAG))

        #validation_loss, validation_accuracy, vizwiz_accuracy, total_sum = 0.0, 0.0, 0.0, 0
        #answerable_true = []
        #answerable_predicted = []
        submission = []
        #ans_type_id = True

        self.eval()
        with torch.no_grad():
            idx=0
            #for _, batch in enumerate(dataloader):
            for batch in tqdm.tqdm(dataloader):
                image, question = batch
                image, question = image.to(device), question.to(device)

                if not dl24_json:
                    output, answer_type_predicted, answerable_predict = self.forward(image, question)
                else:
                    output = self.forward(image, question)

                #print("\t{} : output.size()={}".format(DBG_TAG, output.size()))
                n_batch = output.size()[0]
                if n_batch == 1:
                    pred = output.argmax(1).cpu().item() # スカラー化
                    submission.append(pred)
                else:
                    pred = output.argmax(1)
                    for i in range(n_batch):
                        submission.append(pred[i])                
                idx+=1

        submission = [dataset.idx2answer[id] for id in submission]
        if False:
            print('\t{} : submission1={}'.format(DBG_TAG, submission))
        submission = np.array(submission)
        if False:
            print('\t{} : submission2={}'.format(DBG_TAG, submission))
        np.save("{}{}submission_{}.npy".format(npy_dir, os.sep, filehead), submission)

        return
    
    def test_step(self, dataloader):
        self.eval()
        accuracy, total_sum, vizwiz_accuracy = 0.0, 0, 0.0
        answerable_true = []
        answerable_predicted = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                image, question, answer, answer_type, answers_for_questions, answerable = batch
                image, question, answer, answer_type, answers_for_questions, answerable = image.to(self.device), question.to(self.device), answer.to(self.device), answer_type.to(self.device), answers_for_questions.to(self.device), answerable.to(self.device)
                output, _, answerable_predict = self.forward(image, question)
                answerable = 1 - answerable
                answerable_predict = 1.0 - answerable_predict
                predicted_answer = torch.argmax(output, dim = 1)
                actual_answer = torch.argmax(answer, dim = 1)
                for i in range(len(answer)):
                    if torch.sum(answer[i]) == 0:
                        continue
                    if predicted_answer[i] == actual_answer[i]:
                        accuracy += 1
                    vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                    total_sum +=1
                    answerable_true.append(answerable[i].item())
                    answerable_predicted.append(answerable_predict[i].item())
            
        answerable_true = np.array(answerable_true)
        answerable_predicted = np.array(answerable_predicted)
        
        accuracy /= total_sum
        vizwiz_accuracy /= total_sum
        return accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')

    def save_model(self, path):
        """
        Saves the model state dictionary to the given path.

        Args:
        - self: the model object
        - path (str): the path to save the model state dictionary

        Returns:
        - None
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model state dictionary from the given path.

        Args:
        - self: the model object
        - path (str): the path to load the model state dictionary

        Returns:
        - self: the loaded model object
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        return self

    def predict(self, image, question):
        """
        Predicts the output and answer type for the given image and question.

        Args:
        - self: the model object
        - image (tensor): the image tensor
        - question (tensor): the question tensor

        Returns:
        - output (tensor): the predicted output tensor
        - answer_type (str): the predicted answer type
        """
        output, answer_type, answerability = self.forward(image, question)
        answerability = 1.0 - answerability
        return output, answer_type, answerability

    def plot_loss(self):
        """
        Plots the training and validation losses.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_losses, label = "Training Loss")
        plt.plot(self.validation_losses, label = "Validation Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        """
        Plots the training and validation accuracies.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_accuracies, label = "Training Accuracy")
        plt.plot(self.validation_accuracies, label = "Validation Accuracy")
        plt.legend()
        plt.show()

    def plot_vizwiz_accuracy(self):
        """
        Plots the VizWiz training and validation accuracies.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.vizwiz_training_accuracies, label = "VizWiz Training Accuracy")
        plt.plot(self.vizwiz_validation_accuracies, label = "VizWiz Validation Accuracy")
        plt.legend()
        plt.show()

    def plot_answerability(self):
        """
        Plots the training and validation answerabilities.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_answerability, label = "Training Answerability")
        plt.plot(self.validation_answerability, label = "Validation Answerability")
        plt.legend()
        plt.show()

    def test_model(self, image_path, question):
        """
        Tests the model by predicting the answer and answer type for the given image and question.

        Args:
        - self: the model object
        - image_path (str): the path to the image file or URL
        - question (str): the question to be asked

        Returns:
        - predicted_answer (tensor): the predicted answer tensor
        - predicted_answer_type (str): the predicted answer type
        """
        self.eval()
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream = True).raw)
        else:
            image = Image.open(image_path)

        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image)
        image_features = torch.flatten(image_features, start_dim=1)

        question =  clip.tokenize(question).to(self.device)
        text_features = self.clip_model.encode_text(question).float()
        text_features = torch.flatten(text_features, start_dim=1)

        predicted_answer, predicted_answer_type, answerability = self.predict(image_features, text_features)
        return predicted_answer, predicted_answer_type, answerability

    def print_CLIP_model(self):
        """
        Prints the details of the selected CLIP model.

        Args:
        - self: the model object

        Returns:
        - None
        """
        input_resolution = self.clip_model.visual.input_resolution
        context_length = self.clip_model.context_length
        vocab_size = self.clip_model.vocab_size

        print("Selected model:", self.model_name)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)
        print("")

# ============================================================
# 
# ============================================================
def Loading_Preprocessed_Embeddings(OUTPUT_PATH):

    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)

    print("\t{} : Start".format(DBG_TAG))

    with open(OUTPUT_PATH + 'training_images.pkl', 'rb') as f:
        training_images = pickle.load(f)
    with open(OUTPUT_PATH + 'training_questions.pkl', 'rb') as f:
        training_questions = pickle.load(f)

    with open(OUTPUT_PATH + 'validation_images.pkl', 'rb') as f:
        validation_images = pickle.load(f)
    with open(OUTPUT_PATH + 'validation_questions.pkl', 'rb') as f:
        validation_questions = pickle.load(f)

    with open(OUTPUT_PATH + 'test_images.pkl', 'rb') as f:
        test_images = pickle.load(f)
    with open(OUTPUT_PATH + 'test_questions.pkl', 'rb') as f:
        test_questions = pickle.load(f)

    imgs = [training_images, validation_images, test_images]
    questions = [training_questions, validation_questions, test_questions]

    print("\t{} : End".format(DBG_TAG))

    return imgs, questions
    
# ============================================================
# 
# ============================================================
def Preparing_Data_Sets(dfs, MODEL_NAME, imgs, questions, KAGL_PATHS, DL24_PATHS, flg_Pickle=True, clip_model=None, preprocessor=None, dl24_json=False, device="cpu"):

    [kagl_train_df, kagl_validation_df, dl24_train_df, dl24_validation_df] = dfs

    '''
    [train_df, validation_df, test_df] = df
    if imgs:
        [training_images, validation_images, test_images] = imgs
    else:
        [training_images, validation_images, test_images] = [None, None, None]
    if questions:
        [training_questions, validation_questions, test_questions] = questions
    else:
        [training_questions, validation_questions, test_questions] = [None, None, None]
    '''

    #[KAGL_train_df, KAGL_validation_df] = KAGL_df
    #[DL24_train_df, DL24_validation_df] = DL24_df
    '''
    if imgs:
        [training_images, validation_images] = imgs
    else:
        [training_images, validation_images] = [None, None]
    if questions:
        [training_questions, validation_questions] = questions
    else:
        [training_questions, validation_questions] = [None, None]
    '''
    #[INPUT_PATH, OUTPUT_PATH, TRAIN_PATH, VALIDATION_PATH, ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH] = PATHS
    [DL24_INPUT_PATH, OUTPUT_PATH, DL24_TRAIN_PATH, DL24_VALIDATION_PATH, DL24_ANNOTATIONS_TRAIN_PATH, DL24_ANNOTATIONS_VAL_PATH] = DL24_PATHS
    [KAGL_INPUT_PATH, OUTPUT_PATH, KAGL_TRAIN_PATH, KAGL_VALIDATION_PATH, KAGL_ANNOTATIONS_TRAIN_PATH, KAGL_ANNOTATIONS_VAL_PATH] = KAGL_PATHS

    #if dl24_json:
    #    train_df = DL24_train_df
    #    df_list = [KAGL_train_df, KAGL_validation_df]
    #else:
    #    train_df = KAGL_train_df
    #    df_list = [KAGL_validation_df, DL24_train_df]
    #for df in df_list:
    #    print(df["answers"])
    #print(validation_df1["answers"])
    #print(train_df2["answers"])
    #print(validation_df2["answers"])
    # Constructing the training dataset
    dfs_for_ansdict = [kagl_train_df, kagl_validation_df, dl24_train_df]
    kagl_training_dataset = VizWizDataset(dfs_for_ansdict, None, None, MODEL_NAME, 
                                     image_path=KAGL_TRAIN_PATH, add_answer_path="{}{}".format(INPUT_PATH, '/class_mapping.csv'),
                                     dl24_json=False, flg_Pickle=flg_Pickle, 
                                     clip_model=clip_model, preprocessor=preprocessor, have_answer=True, device=device)

    dfs_for_ansdict = [dl24_train_df]
    dl24_training_dataset = VizWizDataset(dfs_for_ansdict, None, None, MODEL_NAME, 
                                     image_path=DL24_TRAIN_PATH, add_answer_path="{}{}".format(INPUT_PATH, '/class_mapping.csv'),
                                     dl24_json=True, flg_Pickle=flg_Pickle, 
                                     clip_model=clip_model, preprocessor=preprocessor, have_answer=True, device=device)

    ANSWER_ONEHOTENCODER = kagl_training_dataset.answer_onehotencoder
    if not dl24_json:
        ANSWER_TYPE_ONEHOTENCODER = kagl_training_dataset.answer_type_onehotencoder
    else:
        ANSWER_TYPE_ONEHOTENCODER = None

    # Saving the fitted one hot encoders
    #with open(OUTPUT_PATH + 'answer_onehotencoder.pkl', 'wb') as f:
    #    pickle.dump(ANSWER_ONEHOTENCODER, f)
    #if not dl24_json:
    #    with open(OUTPUT_PATH + 'answer_type_onehotencoder.pkl', 'wb') as f:
    #        pickle.dump(ANSWER_TYPE_ONEHOTENCODER, f)

    # Constructing the validation dataset
    #if not dl24_json:
    #    have_answer = True
    #else:
    #    have_answer = False
    dfs_for_ansdict = [kagl_validation_df]
    kagl_validation_dataset = VizWizDataset(dfs_for_ansdict, ANSWER_TYPE_ONEHOTENCODER, ANSWER_ONEHOTENCODER, MODEL_NAME, 
                                       image_path=KAGL_VALIDATION_PATH,
                                       dl24_json=False, flg_Pickle=flg_Pickle,
                                        clip_model=clip_model, preprocessor=preprocessor, 
                                        have_answer=True, device=device)

    # Constructing the test dataset
    #test_dataset = VizWizDataset(test_df, ANSWER_TYPE_ONEHOTENCODER, ANSWER_ONEHOTENCODER, MODEL_NAME, test_images, test_questions, 
    #                                image_path=TRAIN_PATH, dl24_json=dl24_json, flg_Pickle=flg_Pickle,
    #                                clip_model=clip_model, preprocessor=preprocessor, have_answer=True, device=device)

    dfs_for_ansdict = [dl24_validation_df]
    dl24_test_dataset = VizWizDataset(dfs_for_ansdict, None, None, MODEL_NAME, 
                                       image_path=DL24_VALIDATION_PATH,
                                       dl24_json=True, flg_Pickle=None,
                                        clip_model=clip_model, preprocessor=preprocessor, 
                                        have_answer=False, device=device)

    dl24_training_dataset.update_dict(kagl_training_dataset)
    kagl_validation_dataset.update_dict(kagl_training_dataset)
    dl24_test_dataset.update_dict(kagl_training_dataset)

    #datasets = [training_dataset, validation_dataset, test_dataset]
    datasets = [kagl_training_dataset, kagl_validation_dataset, dl24_training_dataset, dl24_test_dataset]

    return datasets

# ============================================================
# 
# ============================================================
def Preparing_Data_Loaders(datasets, BATCH_SIZE, shuffle=True):

    #[training_dataset, validation_dataset, test_dataset] = datasets
    [training_dataset, validation_dataset] = datasets

    # Configuring the data loaders
    #BATCH_SIZE = 32 # 64 is good too but 32 is better (variance wise)

    # Constructing the training, validation and test data loaders
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    #dataloaders = [training_dataloader, validation_dataloader, test_dataloader]
    dataloaders = [training_dataloader, validation_dataloader]

    return dataloaders

# ============================================================
# 
# ============================================================
def Training(model, datasets, dataloaders, trn_mode=0, dl24_json=False, 
             params_to_update=None, hyp_prms=[50, 5e-4, 0], filehead="", pth_dir="pth", save_every=1, device="cpu"):

    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)
    print("\t{} : Start".format(DBG_TAG))

#    [training_dataset, validation_dataset, test_dataset] = datasets
#    [training_dataloader, validation_dataloader, test_dataloader] = dataloaders
#    [training_dataset, validation_dataset] = datasets
#    [training_dataloader, validation_dataloader] = dataloaders

    [kagl_training_dataset,    kagl_validation_dataset,    dl24_training_dataset] = datasets
    [kagl_training_dataloader, kagl_validation_dataloader, dl24_training_dataloader] = dataloaders

    # Configuring training's hyperparameters
    #NUM_EPOCHS = 50
    #NUM_EPOCHS = 50
    #LR = 5e-4
    #WEIGHT_DECAY = 0
    NUM_EPOCHS, LR, WEIGHT_DECAY = hyp_prms
    #NUM_CLASSES = len(training_dataset.get_answer_vocab())
    #SAVE_PATH = OUTPUT_PATH
    #SAVE_EVERY = 5

    # Initializing the model
    #model = VQAModel(num_classes=NUM_CLASSES, device= DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
    #model.print_CLIP_model()

    # Initializing the loss function and optimizer
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    if params_to_update is not None:
        optimizer = optim.Adam(params=params_to_update, lr=LR, weight_decay = WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)

    # Training the model and plotting the loss and accuracy
    if trn_mode==1:
        print("\t{} : train kagl_validation_dataloader".format(DBG_TAG))
        model.train_model(kagl_validation_dataloader, None, None, 
                          loss_function, optimizer, epochs=NUM_EPOCHS, 
                          dl24_json=False, filehead=filehead, 
                          save_path=pth_dir, save_every=save_every)
    elif trn_mode==2:
        print("\t{} : dl24_training_dataloader".format(DBG_TAG))
        model.train_model(dl24_training_dataloader, None, None, 
                          loss_function, optimizer, epochs=NUM_EPOCHS, 
                          dl24_json=True, filehead=filehead, 
                          save_path=pth_dir, save_every=save_every)
    else:
        print("\t{} : kagl_training_dataloader".format(DBG_TAG))
        model.train_model(kagl_training_dataloader, None, None, 
                          loss_function, optimizer, epochs=NUM_EPOCHS, 
                          dl24_json=False, filehead=filehead, 
                          save_path=pth_dir, save_every=save_every)
    #model.plot_loss()
    #model.plot_accuracy()
    #model.plot_vizwiz_accuracy()
    #model.plot_answerability()

    return

# ============================================================
# 
# ============================================================

def Eval(model, datasets, dataloaders, dl24_json=False, device="cpu"):
    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)
    print("\t{} : Start".format(DBG_TAG))

    #[training_dataset, validation_dataset, test_dataset] = datasets
    #[training_dataloader, validation_dataloader, test_dataloader] = dataloaders
    [kagl_training_dataset,    kagl_validation_dataset,    dl24_training_dataset] = datasets
    [kagl_training_dataloader, kagl_validation_dataloader, dl24_training_dataloader] = dataloaders

    # Configuring training's hyperparameters
    #NUM_EPOCHS = 50
    NUM_EPOCHS = 1
    #LR = 5e-4
    #WEIGHT_DECAY = 0
    #NUM_CLASSES = len(training_dataset.get_answer_vocab())
    #SAVE_PATH = OUTPUT_PATH
    #SAVE_EVERY = 5

    # Initializing the model
    #model = VQAModel(num_classes=NUM_CLASSES, device= DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
    #model.print_CLIP_model()

    # Initializing the loss function and optimizer
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    #if params_to_update is not None:
    #    optimizer = optim.Adam(params=params_to_update, lr=LR, weight_decay = WEIGHT_DECAY)
    #else:
    #    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)

    # Training the model and plotting the loss and accuracy
    model.eval_model(kagl_validation_dataloader, 
                      loss_function, epochs=NUM_EPOCHS, 
                      dl24_json=dl24_json)
    #model.plot_loss()
    #model.plot_accuracy()
    #model.plot_vizwiz_accuracy()
    #model.plot_answerability()

    return

# ============================================================
# 
# ============================================================
def Test(model, dataloader, dataset, dl24_json=False, filehead="", npy_dir="npy", device="cpu"):
    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)
    print("\t{} : Start".format(DBG_TAG))

    #[training_dataset, validation_dataset, test_dataset] = datasets
    #[training_dataloader, validation_dataloader, test_dataloader] = dataloaders
    #[training_dataset, validation_dataset] = datasets
    #[training_dataloader, validation_dataloader] = dataloaders

    # Configuring training's hyperparameters
    #NUM_EPOCHS = 50
    NUM_EPOCHS = 1
    #LR = 5e-4
    #WEIGHT_DECAY = 0
    #NUM_CLASSES = len(training_dataset.get_answer_vocab())
    #SAVE_PATH = OUTPUT_PATH
    #SAVE_EVERY = 5

    # Initializing the model
    #model = VQAModel(num_classes=NUM_CLASSES, device= DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
    #model.print_CLIP_model()

    # Initializing the loss function and optimizer
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    #if params_to_update is not None:
    #    optimizer = optim.Adam(params=params_to_update, lr=LR, weight_decay = WEIGHT_DECAY)
    #else:
    #    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)

    # Training the model and plotting the loss and accuracy
    #model.eval_model(validation_dataloader, 
    #                  loss_function, epochs=NUM_EPOCHS, 
    #                  dl24_json=dl24_json)
    model.save_submission(dataloader, dataset, loss_function,
                        dl24_json=dl24_json, filehead=filehead, npy_dir=npy_dir, device=device)
    #model.plot_loss()
    #model.plot_accuracy()
    #model.plot_vizwiz_accuracy()
    #model.plot_answerability()

    return

# ============================================================
# 単体で実行した場合は以下が実行される
# ============================================================
if __name__ == "__main__":

    DBG_TAG = '[{}]'.format(sys._getframe().f_code.co_name)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('outdir', type=str, default='',  help='')
    parser.add_argument('--dl24', action='store_true', help='DL24 Json')
    parser.add_argument('--dbg', action='store_true', help='Debug mode')
    parser.add_argument('--cpu',  action='store_true', help='CPU, not CUDA')
    parser.add_argument('--nt', action='store_true', help='Not train')
    parser.add_argument('--tr', type=int, default=0,  help='')
    parser.add_argument('--test', action='store_true', help='')
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--nopth', action='store_true', help='')
    parser.add_argument('--ftune', type=int, default=1,  help='') # defaultで最終層のファインチューニング
    args = parser.parse_args()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print('device={}'.format(device))

    FLG_PICKLE = False # True = Image,Quesition特徴量をPCL保存する(Kaggle版。ほぼコード削除済)
    #                    False = PCL保存しない
    DL24_JSON  = args.dl24 # True = JSONファイルを最終課題版にする
    #                        False = Kaggle版
    Q_FEATURE_REV = True # True = Question特徴量を修正版にする
    #                      False = Kaggle版 (動作保証外)

    # DL24
    # Configuring the paths for the dataset
    DL24_INPUT_PATH  = './dl'
    DL24_ANNOTATIONS = DL24_INPUT_PATH
    DL24_TRAIN_PATH  = DL24_INPUT_PATH + '/train'
    DL24_VALIDATION_PATH = DL24_INPUT_PATH + '/valid'
    DL24_ANNOTATIONS_TRAIN_PATH = DL24_ANNOTATIONS + '/train.json'
    DL24_ANNOTATIONS_VAL_PATH   = DL24_ANNOTATIONS + '/valid.json'

    # Kaggle
    # Configuring the paths for the dataset
    #INPUT_PATH = '/kaggle/input/vizwiz-2023-edition'
    #KAGL_INPUT_PATH = './kaggle/input/vizwiz-2023-edition'
    #KAGL_ANNOTATIONS = KAGL_INPUT_PATH + '/Annotations/'
    #KAGL_TRAIN_PATH  = KAGL_INPUT_PATH + '/train/train'
    #KAGL_VALIDATION_PATH = KAGL_INPUT_PATH + '/val/val'
    #KAGL_ANNOTATIONS_TRAIN_PATH = KAGL_ANNOTATIONS + '/train.json'
    #KAGL_ANNOTATIONS_VAL_PATH   = KAGL_ANNOTATIONS + '/val.json'
    KAGL_INPUT_PATH = './kaggle/'
    KAGL_ANNOTATIONS = KAGL_INPUT_PATH
    KAGL_TRAIN_PATH  = KAGL_INPUT_PATH + '/train'
    KAGL_VALIDATION_PATH = KAGL_INPUT_PATH + '/val'
    KAGL_ANNOTATIONS_TRAIN_PATH = KAGL_ANNOTATIONS + '/train.json'
    KAGL_ANNOTATIONS_VAL_PATH   = KAGL_ANNOTATIONS + '/val.json'

    if DL24_JSON:
        INPUT_PATH = DL24_INPUT_PATH
        ANNOTATIONS = DL24_ANNOTATIONS
        TRAIN_PATH = DL24_TRAIN_PATH
        VALIDATION_PATH = DL24_VALIDATION_PATH
        ANNOTATIONS_TRAIN_PATH = DL24_ANNOTATIONS_TRAIN_PATH
        ANNOTATIONS_VAL_PATH = DL24_ANNOTATIONS_VAL_PATH
    else:
        INPUT_PATH = KAGL_INPUT_PATH
        ANNOTATIONS = KAGL_ANNOTATIONS
        TRAIN_PATH = KAGL_TRAIN_PATH
        VALIDATION_PATH = KAGL_VALIDATION_PATH
        ANNOTATIONS_TRAIN_PATH = KAGL_ANNOTATIONS_TRAIN_PATH
        ANNOTATIONS_VAL_PATH = KAGL_ANNOTATIONS_VAL_PATH


    #OUTPUT_PATH = '/kaggle/working/'
    OUTPUT_PATH = './working.01/'
    ANSWER_SPACE = 0 # Will be configured later when we build the vocab using the methodology described in the paper
    MODEL_NAME = "ViT-L/14@336px" # This is the backbone of the CLIP model

    PATHS = [INPUT_PATH, OUTPUT_PATH, TRAIN_PATH, VALIDATION_PATH, ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH]
    DL24_PATHS = [DL24_INPUT_PATH, OUTPUT_PATH, DL24_TRAIN_PATH, DL24_VALIDATION_PATH, DL24_ANNOTATIONS_TRAIN_PATH, DL24_ANNOTATIONS_VAL_PATH]
    KAGL_PATHS = [KAGL_INPUT_PATH, OUTPUT_PATH, KAGL_TRAIN_PATH, KAGL_VALIDATION_PATH, KAGL_ANNOTATIONS_TRAIN_PATH, KAGL_ANNOTATIONS_VAL_PATH]

    if not args.cpu:
        # Using accelerated computing if available
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = "cpu"
    print("\t{} : Device={}".format(DBG_TAG, DEVICE))

    # ------------------------------------------------------------
    #df, ANSWER_SPACE = Create_DafaFrame(ANNOTATIONS_TRAIN_PATH, ANNOTATIONS_VAL_PATH)
    [kagl_train_df, kagl_validation_df], KAGL_ANSWER_SPACE = Create_DafaFrame_2(KAGL_ANNOTATIONS_TRAIN_PATH, KAGL_ANNOTATIONS_VAL_PATH)
    [dl24_train_df, dl24_validation_df], DL24_ANSWER_SPACE = Create_DafaFrame_DL24(DL24_ANNOTATIONS_TRAIN_PATH, DL24_ANNOTATIONS_VAL_PATH)
#    [train_df, validation_df, test_df] = df
    #if DL24_JSON:
    #    df = DL24_df
    #else:
    #    df = KAGL_df
    #[train_df, validation_df] = df
    dfs = [kagl_train_df, kagl_validation_df, dl24_train_df, dl24_validation_df]
    print("\t{} : kagl_train_df.head()={}".format(DBG_TAG, kagl_train_df.head()))
    print("")
    print("\t{} : kagl_train_df.info()={}".format(DBG_TAG, kagl_train_df.info()))

    #for row in train_df['answers']:
    #    for answer_map in row:
    #        print("answer_map={}".format(answer_map))
    #        print("answer_map['answer']={}".format(answer_map['answer']))

    PTH_list = [
                ["./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e6.pth", "_e6"],
                ["./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e5.pth", "_e5"],
                ["./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e4.pth", "_e4"],
                ["./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e3.pth", "_e3"],
                ["./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e2.pth", "_e2"],
                ]

    if not args.dbg:
    #for ele in PTH_list:

        #PTH_FILE = ele[0]
        #e_num    = ele[1]         
        if args.ftune == 9: 
            PTH_FILE = "./kaggle/kaggle_epoch_50.pth"
        else:
            pass
#                PTH_FILE = "./20240714_07_01_B4_C48589/pth/20240714_07_01_B4_C48589_epoch_1.pth"
#                PTH_FILE = "./20240714_07_01_B4_C48589/pth/20240714_07_01_B4_C48589_epoch_2.pth"
#                PTH_FILE = "./20240714_07_01_B4_C48589/pth/20240714_07_01_B4_C48589_epoch_3.pth"

#                PTH_FILE = "./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e4.pth"
#            PTH_FILE = "./20240715_02_05_O_B64_C48589/pth/model_B64_C48589_e4.pth"
#            PTH_FILE = "./20240716_01_01_O_B64_C48589/pth/model_B64_C48589_e1.pth"
            PTH_FILE = "./20240716_01_01_O_B64_C48589/pth/model_B64_C48589_e2.pth"

        e_num    = "_eX"

        # ============================================================
        #explore_3_dataframe(train_df, validation_df, test_df)

        clip_model, preprocessor = clip.load(MODEL_NAME, device = DEVICE)
        print("\t{} : Loading Clip ...".format(DBG_TAG))
        clip_model.eval().requires_grad_(False)
        print("\t{} : Done.".format(DBG_TAG))

        # ============================================================
        #if FLG_PICKLE and not DL24_JSON:
        #    Pickle_DF(train_df, validation_df, test_df, clip_model, preprocessor, TRAIN_PATH, VALIDATION_PATH, OUTPUT_PATH, DEVICE=DEVICE)
        #
        #    imgs, questions = Loading_Preprocessed_Embeddings(OUTPUT_PATH)
        #    [training_images, validation_images, test_images] = imgs
        #    [training_questions, validation_questions, test_questions] = questions
        #else:
        imgs = None
        questions = None
        [training_images, validation_images] = [None, None]
        [training_questions, validation_questions] = [None, None]

        # ============================================================
        datasets = Preparing_Data_Sets(dfs, MODEL_NAME, imgs, questions, KAGL_PATHS, DL24_PATHS, flg_Pickle=FLG_PICKLE,
                                    clip_model=clip_model, preprocessor=preprocessor, 
                                    dl24_json=DL24_JSON, device=DEVICE)
        [kagl_training_dataset, kagl_validation_dataset, dl24_training_dataset, dl24_test_dataset] = datasets

        print("\t{} : len(training_dataset)={}".format(DBG_TAG, len(kagl_training_dataset)))

        print("\t{} : len(kagl_training_dataset.answer2idx)={}".format(DBG_TAG, len(kagl_training_dataset.answer2idx)))
        print("\t{} : len(kagl_validation_dataset.answer2idx)={}".format(DBG_TAG, len(kagl_validation_dataset.answer2idx)))
        print("\t{} : len(dl24_training_dataset.answer2idx)={}".format(DBG_TAG, len(dl24_training_dataset.answer2idx)))
        print("\t{} : len(dl24_test_dataset.answer2idx)={}".format(DBG_TAG, len(dl24_test_dataset.answer2idx)))

        # ============================================================
        BATCH_SIZE = 64
        #dataloaders = Preparing_Data_Loaders(datasets, BATCH_SIZE, shuffle=False)
        #[training_dataloader, validation_dataloader, test_dataloader] = dataloaders
        #[training_dataloader, validation_dataloader] = dataloaders
        kagl_training_dataloader   = DataLoader(kagl_training_dataset,   batch_size=BATCH_SIZE, shuffle=True)
        kagl_validation_dataloader = DataLoader(kagl_validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        dl24_training_dataloader   = DataLoader(dl24_training_dataset,   batch_size=BATCH_SIZE, shuffle=True)
        dl24_test_dataloader       = DataLoader(dl24_test_dataset,       batch_size=1, shuffle=False)

        trn_dataloaders = [kagl_training_dataloader, kagl_validation_dataloader, dl24_training_dataloader]
        trn_datasets    = [kagl_training_dataset,    kagl_validation_dataset,    dl24_training_dataset]

        #print("\t{} : len(dataloaders)={}".format(DBG_TAG, len(dataloaders)))

        # ============================================================
        NUM_CLASSES_1 = len(kagl_training_dataset.get_answer_vocab()) # Kaggle版
        NUM_CLASSES_2 = len(kagl_training_dataset.answer2idx) # Question特徴量修正
        if Q_FEATURE_REV:
            NUM_CLASSES = NUM_CLASSES_2
        else:
            NUM_CLASSES = NUM_CLASSES_1
        print("\t{} : NUM_CLASSES={} <= {}, {}".format(DBG_TAG, NUM_CLASSES, NUM_CLASSES_1, NUM_CLASSES_2))

        # ============================================================
        # 
        # ============================================================
        filehead="B{}_C{}".format(BATCH_SIZE, NUM_CLASSES)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        if not args.dbg:
            #outdir   = "{}{}{}".format(args.outdir, os.sep, filehead)
            outdir   = "{}_{}".format(args.outdir, filehead)
            log_dir  = '{}{}log'.format(outdir, os.sep)
            npy_dir  = '{}{}npy'.format(outdir, os.sep)
            pth_dir  = '{}{}pth'.format(outdir, os.sep)
            dump_dir  = '{}{}dump'.format(outdir, os.sep)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(pth_dir, exist_ok=True)
            os.makedirs(dump_dir, exist_ok=True)

        # ============================================================
        # Initializing the model

        # 最終クラス数
        num_class_kaggle_model = 5239 # KaggleからDLしたモデルを使う場合はこちら

        if args.ftune == 2:
            tmp_num_class = num_class_kaggle_model
        else:
            tmp_num_class = NUM_CLASSES
        print("\t{} : tmp_num_class".format(DBG_TAG, tmp_num_class))

        # 隠れクラス数
        HIDDEN_CLASS  = 512
        #model = VQAModel(num_classes=NUM_CLASSES, device= DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
        model = VQAModel(num_classes=tmp_num_class, device= DEVICE, hidden_size=HIDDEN_CLASS, model_name=MODEL_NAME).to(DEVICE)
        model.print_CLIP_model()

        # ============================================================
        # 1. モデルのパラメータをロード
        # ============================================================

        if not args.nopth:

            print("\t{} : Loading PTH_FILE={}".format(DBG_TAG, PTH_FILE))
            model.load_model(PTH_FILE)

        # ============================================================
        # 2. モデルをファインチューニング・転移学習
        # ============================================================

        print("\t{} : args.ftune={}".format(DBG_TAG, args.ftune))

        if args.ftune > 0: 

            # args.ftune == 0 : 全パラメータを学習
            # args.ftune == 1 : 最終クラスのFine Tuning
            # args.ftune == 2 : 最終クラスの数を変える転移学習 (kaggle版から)
            # args.ftune == 3 : 最終クラスの数を変える転移学習

            print("\t{} : Fine Tunnig at Last Layers !!".format(DBG_TAG))

            if args.ftune >= 2: 

                print("\t{} : Change n_class {} -> {} !!".format(DBG_TAG, tmp_num_class, NUM_CLASSES))

                model.linear_layer2[2] = nn.Linear(HIDDEN_CLASS, NUM_CLASSES)
                model.answer_mask_layer = nn.Linear(4, NUM_CLASSES)
            else:

                print("\t{} : Not change n_class {} -> {} !!".format(DBG_TAG, tmp_num_class, NUM_CLASSES))

            # パラメータ固定
            #for param in model.parameters():
            #    param.requires_grad = False
            params_to_update = []
            for name, param in model.named_parameters():
                #print("name={}, param.size()={}".format(name, param.size()))
                if "linear_layer2.2" in name or "answer_mask_layer" in name:
                    print("\t{} : params_to_update : name={}, param.size()={}".format(DBG_TAG, name, param.size()))
                    #print("name={}, param={}".format(name, param))
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
            #print("params_to_update={}".format(params_to_update))
            #print(model)

            model = model.to(DEVICE)

        else:

            params_to_update = None
            print("\t{} : NOT fine tuning".format(DBG_TAG))

        # ============================================================
        # 課題提出用ptyの再生
        # ============================================================
        if args.test:

            filehead_test = "{}_{}_{}".format(args.outdir, filehead, e_num)

            Test(model, dl24_test_dataloader, dl24_test_dataset, dl24_json=DL24_JSON, filehead=filehead_test, 
                npy_dir=npy_dir, device=DEVICE)

        # ============================================================
        elif args.eval:

            Eval(model, trn_datasets, trn_dataloaders, dl24_json=DL24_JSON, device=DEVICE)
            pass

        # ============================================================
        # args.nt == True : 学習
        # args.tr == 0 : 下記Kaggleとほとんど同じ条件で学習
        # (default)    :  "https://www.kaggle.com/code/abdelghafor/visual-question-answering/notebook"
        #              : ただし大きな仕様変更あり
        # args.tr == 2 : 課題用 train.zip と train.json を使って学習
        #              : FineTuningしてもOKだが、これを使って学習することが課題条件
        elif not args.nt:

            # Warning !! : args.tr == 2

            hyp_prms=[
#                        [50, 5e-4, 0],
#                        [50, 5e-4, 0],
#                        [50, 5e-4, 0],
                        [50, 1e-3, 0],
                        [50, 1e-3, 0],
                        [50, 1e-3, 0],
                    ]

            Training(model, trn_datasets, trn_dataloaders, trn_mode=args.tr, dl24_json=DL24_JSON,
                     params_to_update=params_to_update, hyp_prms=hyp_prms[args.tr], filehead=filehead,
                     pth_dir="{}{}".format(pth_dir, os.sep), save_every=1, device=DEVICE)

        # ============================================================
        else:
            pass

    else:
        pass

# EOF
