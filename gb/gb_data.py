import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader, Dataset
import re

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6


def location_to_country(users):
    # Has location_state but no location_country
    state_to_country = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location_list = set()
    for location in state_to_country:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.add(right_location)
        except:
            pass
    
    for location in location_list:
        users.loc[users[(users['location_state']==location.split(',')[1])&(users['location_country'].isna())].index,'location_country'] = location.split(',')[2]

    # Has location_city but no location_country
    city_to_country = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location_list = set()
    for location in city_to_country:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.add(right_location)
        except:
            pass

    for location in location_list:
        users.loc[users[(users['location_city']==location.split(',')[0])&(users['location_country'].isna())].index,'location_country'] = location.split(',')[2]

    users['location_country'].fillna('unknown',inplace=True)

    return users


def process_gb_data(users, books, ratings1, ratings2):
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '')
    users = users.replace('na', np.nan)
    users = users.replace('', np.nan)

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    
    users = location_to_country(users)
    users = users.drop(['location','location_city','location_state'], axis=1)

    # User - Label encoding
    u_encoder = LabelEncoder()
    u_encoder.fit(users['location_country'])
    users['location_country'] = u_encoder.transform(users['location_country'])

    # Books - Label encoding
    b_label = ['category', 'language', 'book_author'] # 'year_of_publication'
    b_encoder = dict()
    for l in b_label:
        b_encoder[l] = LabelEncoder()
        b_encoder[l].fit(books[l])
        books[l] = b_encoder[l].transform(books[l])

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    # Add or replace with 'year_of_publication'
    exp_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    # loc_city2idx = {v:k for k,v in enumerate(exp_df['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(exp_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(exp_df['location_country'].unique())}

    # train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    # test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # train_df['age'] = train_df['age'].fillna(int(train_df['age'].median()))
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)

    # test_df['age'] = test_df['age'].fillna(int(test_df['age'].median()))
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(exp_df['category'].unique())}
    # publisher2idx = {v:k for k,v in enumerate(exp_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(exp_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(exp_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    # train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    # test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        # "loc_city2idx":loc_city2idx,
        # "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        # "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def gb_data_load(args):
    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    # books 
    books['isbn'] = books['img_url'].apply(lambda x: x.split('P/')[1][:10])
    books['category'] = books['category'].apply(lambda x: re.sub('[\W_]+',' ',str(x)).strip())
    # books.fillna('-1',inplace=True)

    # ratings
    train['rating'].fillna(5,inplace=True)

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    # interaction matrix(train, sub, test)
    # interaction = exp_interaction(idx2user, idx2isbn, train, sub)

    idx, exp_train, exp_test = process_gb_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, # len(idx['loc_city2idx']), 
                            # len(idx['loc_state2idx']), 
                            len(idx['loc_country2idx']),
                            len(idx['category2idx']), 
                            #len(idx['publisher2idx']), 
                            len(idx['language2idx']), 
                            len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':exp_train,
            'test':exp_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


# def gb_interaction(idx2user, idx2isbn, train, sub, test):
#     size_uid = idx2user.keys()
#     size_iid = idx2isbn.keys()

#     ui_shape = (len(size_uid), len(size_iid))

#     user_cat = CategoricalDtype(categories=sorted(size_uid), ordered=True)
#     book_cat = CategoricalDtype(categories=sorted(size_iid), ordered=True)

#     ratings = pd.concat([train, sub, test])
#     user_index = ratings["user_id"].astype(user_cat).cat.codes
#     book_index = ratings["isbn"].astype(book_cat).cat.codes

#     interactions = sparse.coo_matrix((ratings["rating"], (user_index,book_index)), shape=ui_shape)

#     uids, iids, data = shuffle_data(interactions, random_state)
#     train_idx, test_idx = cutoff_by_user(uids, test_percentage)

#     train = sparse.coo_matrix(
#         (data[train_idx], (uids[train_idx], iids[train_idx])),
#         shape=ui_shape,
#         dtype=interactions.dtype,
#     )
    
#     sub = sparse.coo_matrix(
#         (data[sub_idx], (uids[sub_idx], iids[sub_idx])),
#         shape=ui_shape,
#         dtype=interactions.dtype,
#     )

#     test = sparse.coo_matrix(
#         (data[test_idx], (uids[test_idx], iids[test_idx])),
#         shape=ui_shape,
#         dtype=interactions.dtype,

#     return train, sub, test


def gb_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

