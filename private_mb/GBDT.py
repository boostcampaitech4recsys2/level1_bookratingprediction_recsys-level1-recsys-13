from pandas.api.types import CategoricalDtype
from scipy import sparse


# import
books = pd.read_csv('./books.csv')
users = pd.read_csv('./users.csv')
ratings = pd.read_csv('./ratings.csv')

# preprocessing
books['isbn'] = books['img_url'].apply(lambda x: x.split('P/')[1][:10])

user2idx = {v:k for k,v in enumerate(ratings['user_id'].unique())}
book2idx = {v:k for k,v in enumerate(ratings['isbn'].unique())}

ratings['iid'] = ratings['isbn'].map(book2idx)
ratings['uid'] = ratings['user_id'].map(user2idx)

books.fillna(-1,inplace=True)
users.fillna(-1,inplace=True)
ratings.fillna(-1,inplace=True)


# interaction matrix 
size_uid = ratings["uid"].unique()
size_iid = ratings["iid"].unique()

ui_shape = (len(size_uid), len(size_iid))

user_cat = CategoricalDtype(categories=sorted(size_uid), ordered=True)
book_cat = CategoricalDtype(categories=sorted(size_iid), ordered=True)

user_index = ratings["uid"].astype(user_cat).cat.codes
book_index = ratings["iid"].astype(book_cat).cat.codes

interactions = sparse.coo_matrix((ratings["rating"], (user_index,book_index)), shape=ui_shape)


def shuffle_data(interactions:sparse.coo_matrix, random_state:int=42)->tuple:
    random_state = np.random.RandomState(seed=random_state)

    interactions = interactions.tocoo()

    uids, iids, data = (interactions.row, interactions.col, interactions.data)

    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)

    uids = uids[shuffle_indices]
    iids = iids[shuffle_indices]
    data = data[shuffle_indices]

    return uids, iids, data

# don't need -> get indices of train and test data
def cutoff_by_user(uids:list, test_percentage:float=0.2):
    cutoff = int((1.0 - test_percentage) * len(uids))
    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)
    return train_idx, test_idx

def random_train_test_split(interactions, test_percentage=0.2, valid=None, random_state=42):
    uids, iids, data = shuffle_data(interactions, random_state)
    train_idx, test_idx = cutoff_by_user(uids, test_percentage)
    shape = interactions.shape

    train = sparse.coo_matrix(
        (data[train_idx], (uids[train_idx], iids[train_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )
    test = sparse.coo_matrix(
        (data[test_idx], (uids[test_idx], iids[test_idx])),
        shape=shape,
        dtype=interactions.dtype,
    )
    return train, test

train, test = random_train_test_split(interactions,test_percentage=0.2,random_state=42)
train = train.tocsr()
test = test.tocsr()

# Dataframe
uids, iids, data = shuffle_data(interactions)
train_idx, test_idx = cutoff_by_user(uids)

train_df = pd.DataFrame({'uid':uids[train_idx], 'iid':iids[train_idx], 'ratings':data[train_idx]})
test_df = pd.DataFrame({'uid':uids[test_idx], 'iid':iids[test_idx], 'ratings':data[test_idx]})


# feature engineering
users_location = pd.concat(
    [users.drop('location', axis=1), 
     (users['location'].str.split(',', expand=True)
     .assign(location_country = lambda x : np.where(x[2]=='', x[3], x[2]),
            location_state = lambda x : x[1],
            location_city = lambda x : x[0],
            ) 
      )[['location_country', 'location_state', 'location_city']]
     ], axis=1)

city_tab = pd.DataFrame([['iowa city','iowa','usa'],
['somerset', 'somerset','england'],
['milford','massachusetts','usa'],
['rockvale','tennessee','usa'],
['bronx','newyork','usa'],
['tustin','california','usa'],
['choctaw','choctaw','usa'],
['richmond hill','richmond hill','canada'],
['kuala lumpur','kuala lumpur','malaysia']])
city_tab.columns = ['city','state','country']

for _,row in city_tab.iterrows():
    location_idx = users_location['location_city'] == row.city
    users_location.loc[location_idx,'location_city'] = row.city
    users_location.loc[location_idx,'location_state'] = row.state
    users_location.loc[location_idx,'location_country'] = row.country

users_location.fillna('unknown',inplace=True)
users_location['location_country'] = users_location['location_country'].str.replace('n/a','unknown')
users_location['location_city'] = users_location['location_city'].str.replace('n/a','unknown')
users_location['location_state'] = users_location['location_state'].str.replace('n/a','unknown')

users_df = users_location.copy()
users_df['uid'] = users_df['user_id'].map(user2idx)

u_label = ['location_country', 'location_state', 'location_city']
u_encoder = dict()
for l in u_label:
    u_encoder[l] = LabelEncoder()
    u_encoder[l].fit(users_df[l])
    users_df[l] = u_encoder[l].transform(users_df[l])

# categorize age
users_df['age'] = users_df['age'].map(lambda x:x//10 if type(x)!=str else -1)


## Item - label encoding
books_df = books.copy()
books_df['iid'] = books_df['isbn'].map(book2idx)

books_df['category'] = books_df['category'].apply(lambda x: re.sub('[\W_]+',' ',str(x)).strip()) # clean

# 결측치 처리 시, -1로 대체하였기 때문에 label encoding의 작동을 원활히 하기 위해 데이터의 type을 일치시켜
books_df['category'] = books_df['category'].astype(str)
books_df['language'] = books_df['language'].astype(str)

# year_of_publication , language, category를 제외하고 drop합니다. (maybe put author instead of year of publication)
i_label = ['category', 'language']
i_encoder = dict()
for l in i_label:
    i_encoder[l] = LabelEncoder()
    i_encoder[l].fit(books_df[l])
    books_df[l] = i_encoder[l].transform(books_df[l])


# 각각의 데이터 프레임을 iid, uid를 기준으로 merge해 줍니다.
train_context = train_df.merge(users_df, on='uid', how='left').merge(books_df, on='iid', how='left')
test_context = test_df.merge(users_df, on='uid', how='left').merge(books_df, on='iid', how='left')

# 분할 - 다른 모델과 결과 공유할 때 같은 uid, iid 임을 확인하기 위해 rating_train/test 를 저장합니다.
X_train = train_context.drop(['user_id', 'isbn', 'iid','uid', 'ratings'], axis=1)
y_train = train_context['ratings']
rating_train = train_context[['uid','iid','ratings']] 

X_test = test_context.drop(['user_id', 'isbn', 'iid','uid', 'ratings'], axis=1)
y_test = test_context['ratings']
rating_test = test_context[['uid','iid','ratings']] 

# 불필요한 feature를 제거하여 줍니다.
drop_lst = ['book_title','book_author','publisher','img_url','img_path','summary']

X_train.drop(drop_lst,axis=1,inplace=True)
X_test.drop(drop_lst,axis=1,inplace=True)



# RMSE fn.
def modify_range(rating):
  if rating < 0:
    return 0
  elif rating > 10:
    return 10
  else:
    return rating

def rmse(real, predict):
  pred = list(map(modify_range, predict))  
  pred = np.array(pred)
  return np.sqrt(np.mean((real-pred) ** 2))

def matrix_rmse(real_mat, predict_mat, test_ind=test_idx):
  cost = 0

  for i, ind in enumerate(test_ind):
    pred = predict_mat[ind]
    real = real_mat[ind]
    cost += pow(real - modify_range(pred), 2)
  return np.sqrt(cost/ len(test_ind)) 




# Combine feature