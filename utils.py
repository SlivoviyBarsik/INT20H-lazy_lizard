import pandas as pd 
import numpy as np


from sklearn.metrics.pairwise import rbf_kernel

def apply_rbf(X):
    return rbf_kernel(X, [[450]], gamma=0.00002) + rbf_kernel(X, [[1050]], gamma=0.000006)

def load_data(orders_path, nodes_path, train = True):
    orders = pd.read_csv(orders_path)
    nodes  = pd.read_csv(nodes_path)

    nodes = nodes[nodes["Id"].isin(orders.Id)]

    #Fill NA with mean
    nodes.speed.fillna(nodes.speed.mean(), inplace=True)

    nodes.speed = nodes.speed / 3.6 # From km/h to m/s
    nodes['expected_node_time'] = nodes.distance / nodes.speed

    orders = orders.merge(nodes.groupby('Id').sum()['expected_node_time'], left_on='Id', right_index=True).rename({'expected_node_time': 'expected_time'}, axis=1)

    orders.running_time = pd.to_datetime(orders.running_time)
    orders.running_time = orders.running_time.dt.hour * 60 + orders.running_time.dt.minute

    if train:
        orders.completed_time = pd.to_datetime(orders.completed_time)   
        orders.completed_time = orders.completed_time.dt.hour * 60 + orders.completed_time.dt.minute


    orders = orders.sort_values('running_time')

    if train:
        orders.loc[(orders['running_time'] > 1300) & (orders['completed_time'] < 1000), 'completed_time'] = 1440 + orders.loc[(orders['running_time'] > 1300) & (orders['completed_time'] < 1000), 'completed_time']

    return orders, nodes

def feat_eng(orders, nodes, train_orders = None, X_mean = None, X_std = None, scale = False, train = True):

    # Using the information about latest 10 rides (at the time of a ride start)
    # Comparing the true target and the expected target calculated using distance/speed from nodes file
    # This will improve the prediction of speed

    if train_orders is None: # for testing take data from the training dataset
        train_orders = orders

    latest10_list = []

    for ind, row in orders.iterrows():    
        latest10 = train_orders.loc[train_orders.completed_time < row.running_time].tail(10).mean()
        latest10_value = (latest10.delta_time - latest10.expected_time) / latest10.delta_time + 1
        latest10_list.append(latest10_value)


    latest10_list[:10] = [1] * 10
    orders['latest10'] = latest10_list
    orders['expected_time'] = orders.latest10 * orders.expected_time

    df = pd.merge(orders, nodes, left_on='Id', right_on='Id')

    # One order has 0 distance, removed it from the set
    df = df.loc[df['route_distance_km'] != 0]

    df['node_part_time'] = df['expected_node_time'] / df['expected_time']
    df['node_part_distance'] = df['distance'] / df['route_distance_km'] / 1000

    df['current_time'] = df.groupby('Id').expected_node_time.transform(np.cumsum) / 60 + df.running_time

    df['time_rbf'] = apply_rbf(df[['current_time']])

    if train:
        df['delta_time'] = df['delta_time'] * df['node_part_time']
    
    df.set_index('Id', inplace=True)

    X = df[['running_time', 'route_distance_km', 'latest10', 'distance', 'speed', 'expected_node_time', 'node_part_time', 'node_part_distance', 'time_rbf']]
    
    if train:
        y = df['delta_time']
    else:
        y = None


    if X_mean is None or X_std is None:
        X_mean = X.mean(axis = 0)
        X_std = X.std(axis = 0)

    X_scaled = (X - X_mean) / X_std

    if scale:
        return X_scaled, y, X_mean, X_std
    
    else:
        return X, y
    

def train_test_split(X, y, size = 1000):
    test_ind = np.random.choice(X.index.unique(), size=size, replace=False)
    X_test = X.loc[test_ind]
    y_test = y.loc[test_ind]
    X_train = X.loc[~X.index.isin(test_ind)]
    y_train = y.loc[~X.index.isin(test_ind)]

    return X_train, y_train, X_test, y_test

def get_eta_from_nodes(y_nodes):
    return y_nodes.groupby('Id').sum()

