import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def make_plot(y_true, y_pred, title):
    '''
    Creates and labels the plot given the input.
    '''
    
    plt.scatter(y_true, y_pred, alpha = 0.4, c = 'blue', label = title)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.legend()
    plt.savefig('./plots/' + title + '.png')
    plt.show()
    plt.clf()


# Using root mean squared error as my score metric.
def score(y_pred, y_true):
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rounded_rmse = round(rmse, 3)
    
    return rounded_rmse


# Used to get the train set and test set from the DataFrame as input.
def gather_data(pokemon, stat):
    
    x = pokemon.drop([stat], axis = 1)
    y = pokemon[stat]
    
    return train_test_split(x, y, test_size = 0.20, random_state = 42)



def random_forest_regression(pokemon, stat):
    '''
    Section for using the Random Forest Regressor model to fit the data.
    '''
    
    x_train, x_test, y_train, y_test = gather_data(pokemon, stat)

    rfr_model = RandomForestRegressor()
    
    rfr_model.fit(x_train, y_train)
    
    y_pred = rfr_model.predict(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print('\n Random Forest Regression: ' + stat + ' - ' + str(accuracy))
    
    make_plot(y_test, y_pred, 'Random Forest Regression - ' + stat)


def linear_regression(pokemon, stat):
    '''
    Section for using the Linear Regression model to fit the data.
    '''
    
    x_train, x_test, y_train, y_test = gather_data(pokemon, stat)

    lr_model = LinearRegression()
    
    lr_model.fit(x_train, y_train)
    
    y_pred = lr_model.predict(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print('\n Linear Regression: ' + stat + ' - ' + str(accuracy))
    
    make_plot(y_test, y_pred, 'Linear Regression - ' + stat)


def decision_tree_regression(pokemon, stat):
    '''
    Section for using the Decision Tree Regressor model to fit the data.
    '''
    
    x_train, x_test, y_train, y_test = gather_data(pokemon, stat)
    
    accuracies = []
    
    for i in range(1,15):
        
        dtr_model = DecisionTreeRegressor(max_depth = i)
        
        dtr_model.fit(x_train, y_train)
        
        y_pred = dtr_model.predict(x_test)
        
        accuracy = score(y_pred, y_test)
        accuracies.append(accuracy)
        
        if i == 1:
            
            min_index = i
            best_y_pred = y_pred
        
        elif accuracy < accuracies[min_index]:
            
            min_index = i
            best_y_pred = y_pred
    
    print('\n Decision Tree Regression (Max Depth = ' + str(min_index) + '): ' + stat + ' - ' + str(accuracies[min_index]))
    
    make_plot(y_test, best_y_pred, 'Decision Tree Regression - ' + stat)


def support_vector_regression(pokemon, stat):
    '''
    Section for using the Support Vector Regression model to fit the data.
    '''
    
    x_train, x_test, y_train, y_test = gather_data(pokemon, stat)

    svr_model = make_pipeline(StandardScaler(),
                              SVR(C = 1.0, epsilon = 0.2))
    
    svr_model.fit(x_train, y_train)
    
    y_pred = svr_model.predict(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print('\n Support Vector Regression: ' + stat + ' - ' + str(accuracy))
    
    make_plot(y_test, y_pred, 'Support Vector Regression - ' + stat)
    
    
def lasso_regression(pokemon, stat):
    '''
    Section for using the Lasso Regression model to fit the data.
    '''
    
    x_train, x_test, y_train, y_test = gather_data(pokemon, stat)

    lasso_model = linear_model.Lasso(alpha = 0.1)
    
    lasso_model.fit(x_train, y_train)
    
    y_pred = lasso_model.predict(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print('\n Lasso Regression: ' + stat + ' - ' + str(accuracy))
    
    make_plot(y_test, y_pred, 'Lasso Regression - ' + stat)


def main():
    
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    pd.set_option('display.max_columns', 15)
    
    pokemon = pd.read_csv('./csv/cleaned_pokemon.csv')
    
    stats = ['hp', 'attack', 'defense', 'spatk', 'spdef', 'speed']
    
    pokemon = pokemon[stats]
    
    for stat in stats:
        
        print('\n\n')
        random_forest_regression(pokemon, stat)
        linear_regression(pokemon, stat)
        decision_tree_regression(pokemon, stat)
        support_vector_regression(pokemon, stat)
        lasso_regression(pokemon, stat)
    

if __name__ == '__main__':
    
    main()