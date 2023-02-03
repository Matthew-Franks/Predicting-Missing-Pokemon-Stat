import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings


# Used to get the train set and test set from the DataFrame as input.
def gather_data(pokemon):
    
    pokemon.columns = pokemon.columns.str.lower()
    pokemon.columns = pokemon.columns.str.replace('.', '')
    pokemon.columns = pokemon.columns.str.replace('\s', '')

    pokemon = pokemon.drop(['#'], axis = 1)
    pokemon['name'] = pokemon['name'].str.lower()
    
    return pokemon


def remove_mega(pokemon):
    
    pokemon['name'] = pokemon['name'].str.replace('.*(?=mega )', '')
    
    return pokemon


def fill_type2_nan(pokemon):
    
    pokemon['type2'].fillna(pokemon['type1'], inplace = True)

    return pokemon


def pprint(p):
    
    print()
    print(p)


def print_information(pokemon):
    
    pprint(pokemon.columns)
    pprint(pokemon.shape)
    pprint(pokemon.head())


def print_pokemon_describe(pokemon):
    
    pprint(pokemon.describe().round(2))


def print_pokemon_information(pokemon, name):
    
    pprint(pokemon.loc[name])


def print_strongest_pokemon(pokemon, stats):
    
    pprint('Strongest Pokemon In Each Stat')
    
    for stat in stats:
        
        max_stat = pokemon[stat].max()
        names = pokemon.loc[pokemon[stat] == max_stat].index
        
        print()
        
        for name in names:
            
            print(stat + ': ' + name + ' - ' + str(max_stat))


def print_strongest_pokemon_each_type(pokemon):
    
    unique_types = pokemon['type1'].unique()
    pprint('Strongest Pokemon In Each Type (type1)')
    
    for unique_type in unique_types:
        
        max_total = pokemon.loc[pokemon['type1'] == unique_type]['total'].max()
        names = pokemon.loc[(pokemon['type1'] == unique_type) & (pokemon['total'] == max_total)].index
        
        print()
        
        for name in names:
            
            print(unique_type + ': ' + name + ' - ' + str(max_total))


def print_strongest_pokemon_each_type_no_legendary(pokemon):
    
    unique_types = pokemon['type1'].unique()
    pprint('Strongest Pokemon In Each Type (type1) (No Legendaries)')
    
    for unique_type in unique_types:
        
        max_total = pokemon.loc[(pokemon['type1'] == unique_type) & (pokemon['legendary'] == False)]['total'].max()
        names = pokemon.loc[(pokemon['type1'] == unique_type) & (pokemon['legendary'] == False) & (pokemon['total'] == max_total)].index
        
        print()
        
        for name in names:
            
            print(unique_type + ': ' + name + ' - ' + str(max_total))


def print_strongest_pokemon_each_type_no_legendary_or_mega(pokemon):
    
    unique_types = pokemon['type1'].unique()
    pprint('Strongest Pokemon In Each Type (type1) (No Legendaries or Megas)')
    
    for unique_type in unique_types:
        
        max_total = pokemon.loc[(pokemon['name'].str.match(r'^(mega )') == False) & (pokemon['type1'] == unique_type) & (pokemon['legendary'] == False)]['total'].max()
        names = pokemon.loc[(pokemon['name'].str.match(r'^(mega )') == False) & (pokemon['type1'] == unique_type) & (pokemon['legendary'] == False) & (pokemon['total'] == max_total)]['name']
        
        print()
        
        for name in names:
            
            print(unique_type + ': ' + name + ' - ' + str(max_total))


def plot_stat_value(pokemon, stat, max_value):
    
    if stat == 'total':
        
        bins = range(150, 800, 20)
        
    else:    
        
        bins = range(0, 200, 20)
        
    plt.hist(pokemon[stat], bins = bins, histtype = "bar", color = 'green')
    
    plt.xlabel(stat)
    plt.ylabel('count')
    
    plt.plot()
    
    plt.axvline(pokemon[stat].mean(), linestyle = 'dashed', color = 'red')
    
    plt.show()


def plot_type_information(pokemon, type):
    
    unique_types = pokemon[type].unique()
    values = dict(pokemon[type].value_counts())
    
    values_ordered = []
    
    for unique_type in unique_types:
        
        values_ordered.append(values[unique_type])
        
    plt.pie(values_ordered, labels = unique_types)

    plt.axis('equal')
    
    plt.title('Different Types of Pokemon - ' + type + '\n')
    
    plt.plot()
    
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    
    plt.show()


def plot_stats_by_type(pokemon, stat, type):
    
    plt.subplots(figsize = (15, 5))
    plt.title(stat + ' by ' + type)
    
    sns.boxplot(x = type, y = stat, data = pokemon)
    
    if stat == 'total':
        
        plt.ylim(150, 800)
        
    else:    
        
        plt.ylim(0, 200)
    
    plt.show()
    
    
def plot_strongest_generation(pokemon):
    
    plt.subplots(figsize = (15, 5))
    plt.title('Generation By Total')
    
    sns.boxplot(x = 'generation', y = 'total', data = pokemon)
    
    plt.show()


def main():
    
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    pd.set_option('display.max_columns', 15)
    
    pokemon = pd.read_csv('./csv/pokemon.csv')
    
    print_information(pokemon)
    '''
    pokemon = gather_data(pokemon)
    pokemon = remove_mega(pokemon)
    pokemon = fill_type2_nan(pokemon)
    
    pokemon.to_csv('./csv/cleaned_pokemon.csv', index = False)
    '''
    pokemon_no_name_index = pd.read_csv('./csv/cleaned_pokemon.csv')
    pokemon = pokemon_no_name_index.set_index('name')
    stats = ['hp', 'attack', 'defense', 'spatk', 'spdef', 'speed', 'total']
    
    print_information(pokemon)
    print_pokemon_information(pokemon, 'celebi')
    print_pokemon_describe(pokemon)
    print_strongest_pokemon(pokemon, stats)
    print_strongest_pokemon_each_type(pokemon)
    print_strongest_pokemon_each_type_no_legendary(pokemon)
    print_strongest_pokemon_each_type_no_legendary_or_mega(pokemon_no_name_index)

    
    for stat in stats:
        
        plot_stat_value(pokemon, stat, pokemon[stat].max() + 10)
    
    plot_type_information(pokemon, 'type1')
    plot_type_information(pokemon, 'type2')
    plot_strongest_generation(pokemon)
    
    for type in ['type1', 'type2']:
        
        for stat in stats:
    
            plot_stats_by_type(pokemon, stat, type)
    

if __name__ == '__main__':
    
    main()