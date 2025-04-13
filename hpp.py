import os,sys,warnings
if not sys.warnoptions:    
    warnings.simplefilter('ignore')
    
import numpy as np
from numpy.linalg import pinv,inv
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style='whitegrid')
%matplotlib inline

import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin,ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from mllibs.bl_regressor import BR
pi = 4.0*np.arctan(1.0)

color1 = 'darkviolet'
color2 = 'indigo'

from IPython.core.display import display, HTML, Javascript

color_map = ['#FFFFFF','#FF5733']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = '''
div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s;
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s;
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s;
    border-top: 2px solid %s;
}

div.input_prompt {
color: %s;
}

div.output_prompt {
color: %s; 
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s;
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s;
}

.edit_mode div.cell.selected:before {
background: %s;
}

.edit_mode div.cell.selected {
border-color: %s;

}
'''

def to_rgb(h): 
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba, 
                                            main_color,  main_color, prompt, main_color, main_color, 
                                            main_color, main_color))

def nb(): 
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()
# load the dataset
df = pd.read_csv('/kaggle/input/calihouse/housing.csv')
df.info()
df.info()

# Let's show all columns with missing data as well:
df[df.isnull().any(axis=1)] # any missing data in columns

from sklearn.neighbors import KNeighborsRegressor

# function that imputes a dataframe 
def impute_knn(df):
    
    ''' inputs: pandas df containing feature matrix '''
    ''' outputs: dataframe with NaN imputed '''
    # imputation with KNN unsupervised method

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()         # columns w/ nan 
    cols_no_nan = ldf.columns.difference(cols_nan).values     # columns w/o nan 

    for col in cols_nan:                
        imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ldf.dropna()          # all indicies which which have no missing data 
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
    
    return pd.concat([ldf,ldf_putaside],axis=1)

# Call function that imputes missing data
df2 = impute_knn(df)
# looks like we have a full feature matrix
df2.info()

# 70/30 Split should do
trdata,tedata = train_test_split(df2,test_size=0.3,random_state=43)

trdata.hist(bins=60, figsize=(15,9),color=color1);plt.show()

''' Function to plot correlation of features '''
def corrMat(df,id=False):
    
    corr_mat = df.corr().round(2)
    f, ax = plt.subplots(figsize=(6,6))
    mask = np.zeros_like(corr_mat,dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat,mask=mask,vmin=-1,vmax=1,center=0, 
                cmap='plasma',square=False,lw=2,annot=True,cbar=False);plt.show() 
    corrMat(trdata) # plot masked numpy correlation matrix
    
    ''' Draw a Bivariate Seaborn Pairgrid /w KDE density w/ '''
def snsPairGrid(df):

    ''' Plots a Seaborn Pairgrid w/ KDE & scatter plot of df features'''
    g = sns.PairGrid(df,diag_sharey=False)
    g.fig.set_size_inches(14,13)
    g.map_diag(sns.kdeplot, lw=2) # draw kde approximation on the diagonal
    g.map_lower(sns.scatterplot,s=15,edgecolor="k",linewidth=1,alpha=0.4) # scattered plot on lower half
    g.map_lower(sns.kdeplot,cmap='plasma',n_levels=10) # kde approximation on lower half
    plt.tight_layout()
    # Seaborn get a little slow, let's plot some interesting features
tlist = ['median_income','total_rooms','housing_median_age','latitude','median_house_value','population']
snsPairGrid(trdata[tlist]) 
''' Plot Two Geopandas Plots Side by Side '''
# defining a simple plot function, input list containing features of names found in dataframe
def plotTwo(df,lst):
    
    # load california from module, common for all plots
    cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
    cali = cali.assign(area=cali.geometry.area)
    
    # Create a geopandas geometry feature; input dataframe should contain .longtitude, .latitude
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude,df.latitude))
    proj=gcrs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944) # related to view

    ii=-1
    fig,ax = plt.subplots(1,2,figsize=(21,6),subplot_kw={'projection': proj})
    for i in lst:

        ii+=1
        tgdf = gdf.sort_values(by=i,ascending=True) 
        gplt.polyplot(cali,projection=proj,ax=ax[ii]) # the module already has california 
        gplt.pointplot(tgdf,ax=ax[ii],hue=i,cmap='plasma',legend=True,alpha=1.0,s=3) # 
        ax[ii].set_title(i)

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5)
    
    # Call function that plots two geopandas plots 
plotTwo(trdata,['population','median_income'])
plotTwo(trdata,['housing_median_age','median_house_value'])
del trdata['geometry'] # not useful for anything other than gpd visualisation

# trdata_upd : training data w/ removed outliers
maxval2 = trdata['median_house_value'].max() # get the maximum value
trdata_upd = trdata[trdata['median_house_value'] != maxval2] 
tedata_upd = tedata[tedata['median_house_value'] != maxval2]
trdata_upd.hist(bins=60, figsize=(15,9),color=color1);plt.show() # looks like its completely removed.

# Make a feature that contains both longtitude & latitude
trdata_upd['diag_coord'] = (trdata_upd['longitude'] + trdata_upd['latitude'])         # 'diagonal coordinate', works for this coord
trdata_upd['bedperroom'] = trdata_upd['total_bedrooms']/trdata_upd['total_rooms']     # feature w/ bedrooms/room ratio
corrMat(trdata_upd)
# update test data as well
tedata_upd['diag_coord'] = (tedata_upd['longitude'] + tedata_upd['latitude'])
tedata_upd['bedperroom'] = tedata_upd['total_bedrooms']/tedata_upd['total_rooms']     # feature w/ bedrooms/room ratio

# lets plot them as well
plotTwo(trdata_upd,['diag_coord','median_house_value'])
plotTwo(trdata_upd,['bedperroom','median_house_value'])
del trdata_upd['geometry']  # remove gpd geometry features

''' Draw a a single Heatmap using Seaborn '''
def heatmap1(values,xlabel,ylabel,xticklabels,yticklabels,
            cmap='plasma',vmin=None,vmax=None,fmt="%0.2f"):

    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(values, ax=ax,cmap=cmap)
    
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
    ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
    ax.set_xticklabels(xticklabels);ax.set_yticklabels(yticklabels)
    ax.set_title('BR()')
    ax.set_aspect(1)
    
    for p, color, value in zip(img.get_paths(), img.get_facecolors(),img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
        # Model Evaluation w/ Cross Validation
def modelEval(ldf,feature='median_house_value',model_id = 'dummy'):
    
    # Input: Feature & Target DataFrame

    # Split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]     # remove target variable
    
    # Pick Model 
    if(model_id is 'dummy'):    model = DummyRegressor()
    if(model_id is 'br'):    model = BR(verbose=False)  
    if(model_id is 'rf'):    model = RandomForestRegressor(n_estimators=10,random_state=10)
    
    ''' Parameter Based Cross Validation (No Pipeline)'''
#     gscv = GridSearchCV(model,param_grid,cv=5)
#     gscv.fit(X,y)
#     results = pd.DataFrame(gscv.cv_results_)
#     scores = np.array(results.mean_test_score).reshape(7,7)
    
#     # plot the cross validation mean scores
#     heatmap1(scores,xlabel='lamda',xticklabels=param_grid['lamd'],
#                     ylabel='alpha',yticklabels=param_grid['alph'])
    
    ''' Standard Cross Validation '''
    cv_score = np.sqrt(-cross_val_score(model,X,y,cv=5,scoring='neg_mean_squared_error'))
    print("Scores:",cv_score);print("Mean:", cv_score.mean());print("std:", cv_score.std())
    
    # A simple comparison model
modelEval(trdata,model_id='dummy')
# Original Features
modelEval(trdata,model_id='br')
# Extra Features
modelEval(trdata_upd,model_id='br')

# lets remove two of the three similar features
del trdata_upd['total_bedrooms']
del trdata_upd['total_rooms']

modelEval(trdata_upd,model_id='br')
    
    ''' Plot Two Seaborn Heatmaps Side by Side '''
# used for Polynomial vs non polynomial cross validaion score comparison
def heatmap2(values,values2,xlabel,ylabel,xticklabels,yticklabels,
			cmap='plasma',vmin=None,vmax=None,fmt="%0.2f"):

	fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
	sns.heatmap(values, ax=ax1,cmap=cmap)
	sns.heatmap(values2, ax=ax2,cmap=cmap)
	
	img = ax1.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
	img.update_scalarmappable()
	ax1.set_xlabel(xlabel);ax1.set_ylabel(ylabel)
	ax1.set_xticks(np.arange(len(xticklabels)) + 0.5)
	ax1.set_yticks(np.arange(len(yticklabels)) + 0.5)
	ax1.set_xticklabels(xticklabels);ax1.set_yticklabels(yticklabels)
	ax1.set_title('PolynomialFeatures(2) + BR()')
	ax1.set_aspect(1)
	
	for p, color, value in zip(img.get_paths(), img.get_facecolors(),img.get_array()):
		x, y = p.vertices[:-2, :].mean(0)
		if np.mean(color[:3]) > 0.5:
			c = 'k'
		else:
			c = 'w'
		ax1.text(x, y, fmt % value, color=c, ha="center", va="center")
		
	img = ax2.pcolor(values2, cmap=cmap, vmin=vmin, vmax=vmax)
	img.update_scalarmappable()
	ax2.set_xlabel(xlabel);ax2.set_ylabel(ylabel)
	ax2.set_xticks(np.arange(len(xticklabels)) + 0.5)
	ax2.set_yticks(np.arange(len(yticklabels)) + 0.5)
	ax2.set_xticklabels(xticklabels);ax2.set_yticklabels(yticklabels)
	ax2.set_title('PolynomialFeatures(3) + BR()')
	ax2.set_aspect(1)
	
	for p, color, value in zip(img.get_paths(), img.get_facecolors(),img.get_array()):
		x, y = p.vertices[:-2, :].mean(0)
		if np.mean(color[:3]) > 0.5:
			c = 'k'
		else:
			c = 'w'
		ax2.text(x, y, fmt % value, color=c, ha="center", va="center")
  
  # Model Evaluation Function w/ Pipelines
def modelEval2(ldf,feature='median_house_value',model_id = 'dummy',scaling_id=False):

    # Given a dataframe, split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]     # remove target variable
    
    tlst = []
    for i in [2,3]:
        
        # Pick Model 
        if(model_id is 'dummy'):    model = DummyRegressor()
        if(model_id is 'br'):    model = BR(verbose=False)  
        if(model_id is 'rf'):    model = RandomForestRegressor(n_estimators=10,random_state=10)

        # Pick a Pipeline (Polynomial Feature Adjustment + Model)
        if(scaling_id is False):
            pipe = Pipeline(steps=[('poly',PolynomialFeatures(i)),
                                   ('model',model)])
        else:
            pipe = Pipeline(steps=[('scaler',StandardScaler()),
                                   ('poly',PolynomialFeatures(i)),
                                   ('model',model)])

        ''' Parameter Based Cross Validation (With Pipeline)'''
        # define a parameter search grid, pipepines require slightly different notations w/ __
#         param_grid = {
#         'model__lamd': [0.0001,0.001, 0.01, 0.1, 1, 10, 100],
#         'model__alph': [0.0001,0.001, 0.01, 0.1, 1, 10, 100]}
        
#         gscv2 = GridSearchCV(pipe, param_grid,cv=5)
#         gscv2.fit(X,y)
#         ypred = gscv2.predict(X)
#         results2 = pd.DataFrame(gscv2.cv_results_)
#         scores2 = np.array(results2.mean_test_score).reshape(7,7)
#         tlst.append(scores2)
        
        ''' Standard Cross Validation '''
        cv_score = np.sqrt(-cross_val_score(pipe,X,y,cv=5,scoring='neg_mean_squared_error'))
        print("Scores:",cv_score.round(2))
        print("Mean:", cv_score.mean().round(2));print("std:", cv_score.std().round(2))
    
#     plot mean of 5 cross validation segment score
#     heatmap2(tlst[0],tlst[1],xlabel='lamd', xticklabels=param_grid['model__lamd'],
#                              ylabel='alph', yticklabels=param_grid['model__alph'])
modelEval2(trdata_upd,model_id='br',scaling_id=False)
modelEval2(trdata_upd,model_id='br',scaling_id=True)
modelEval(trdata_upd,model_id='rf')