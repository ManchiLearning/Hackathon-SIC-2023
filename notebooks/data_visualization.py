import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error




# 1. ¿Cuál es la distribución de salarios en diferentes regiones y categorías de trabajo?
def plot_salary_distribution_box(data):
    plt.figure(figsize=(24, 10))
    sns.boxplot(x='region', y='salary_in_usd_2023', hue='job_category', data=data)
    plt.title('Distribución de salarios en diferentes regiones y categorías de trabajo')
    plt.xlabel('Región')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(rotation=45)
    plt.legend(title='Categoría de trabajo', loc='upper right')
    plt.show()

def plot_salary_distribution_box_v2(data):
    plt.figure(figsize=(24, 10))
    sns.boxplot(x='job_category', y='salary_in_usd_2023', hue='region', data=data)
    plt.title('Distribución de salarios en diferentes regiones y categorías de trabajo')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(rotation=45)
    plt.legend(title='Región', loc='upper right')
    plt.show()

# 2. ¿Cómo varía el salario según el nivel de experiencia y el tamaño de la empresa?
def plot_salary_by_experience_and_company_size(data):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='experience_level', y='salary_in_usd_2023', hue='company_size', data=data)
    plt.title('Variación del salario según el nivel de experiencia y tamaño de la empresa')
    plt.xlabel('Nivel de experiencia')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.legend(title='Tamaño de la empresa', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 3. ¿Cuáles son las modalidades de trabajo (remoto, en persona, híbrido) más comunes?
def plot_work_setting_distribution_percentage(data):
    # Calcular el porcentaje de cada modalidad de trabajo
    work_setting_counts = data['work_setting'].value_counts(normalize=True) * 100
    work_setting_percentage = work_setting_counts.reset_index()
    work_setting_percentage.columns = ['work_setting', 'percentage']
    
    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    sns.barplot(data=work_setting_percentage, x='work_setting', y='percentage')
    plt.title('Distribución de modalidades de trabajo (en porcentaje)')
    plt.xlabel('Modalidad de trabajo')
    plt.ylabel('Porcentaje')
    plt.show()


# 4. ¿Cuáles son las modalidades de trabajo más comunes en cada región?
def plot_work_setting_by_region_percentage(data):
    # Calcular el porcentaje de cada modalidad de trabajo en cada región
    region_work_setting_counts = data.groupby('region')['work_setting'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico
    region_work_setting_counts.plot(kind='bar', stacked=True, figsize=(16, 10))
    plt.title('Distribución de modalidades de trabajo por región (en porcentaje)')
    plt.xlabel('Región')
    plt.ylabel('Porcentaje')
    plt.legend(title='Modalidad de trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# 5. ¿Cómo varían las modalidades de trabajo según la categoría de trabajo?
def plot_work_setting_by_job_category_percentage(data):
    # Calcular el porcentaje de cada modalidad de trabajo en cada categoría de trabajo
    job_category_work_setting_counts = data.groupby('job_category')['work_setting'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico
    ax = job_category_work_setting_counts.plot(kind='bar', stacked=True, figsize=(16, 10))
    plt.title('Distribución de modalidades de trabajo por categoría de trabajo (en porcentaje)')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Porcentaje')
    plt.legend(title='Modalidad de trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 6. ¿Cómo influye la modalidad de trabajo en el salario?
def plot_salary_by_work_setting(data):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='work_setting', y='salary_in_usd_2023', data=data)
    plt.title('Distribución de salarios según la modalidad de trabajo')
    plt.xlabel('Modalidad de trabajo')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.show()



# 7. ¿Qué regiones tienen la mayor demanda de empleos en la industria de los datos?
def plot_job_demand_by_region_heatmap(data):
    # Calcular el conteo de empleos por región
    region_counts = data['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    
    # Crear el heatmap
    plt.figure(figsize=(14, 8))
    heatmap_data = region_counts.set_index('region').T
    sns.heatmap(data=heatmap_data, annot=True, fmt="d", cmap='viridis', linewidths=.5)
    plt.title('Demanda de empleos en la industria de los datos por región')
    plt.xlabel('Región')
    plt.ylabel('Conteo de Empleos')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()






# 8. ¿Existe una correlación significativa entre la localización y el salario?
def plot_salary_by_region(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region', y='salary_in_usd_2023', data=data)
    plt.title('Distribución de salarios según la región')
    plt.xlabel('Región')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 9. ¿Cuál es la distribución de los tipos de empleo (tiempo completo, medio tiempo, contrato)?

def plot_employment_type_distribution_heatmap(data):
    # Calcular el conteo de cada tipo de empleo
    employment_type_counts = data['employment_type'].value_counts().reset_index()
    employment_type_counts.columns = ['employment_type', 'count']
    
    # Crear un DataFrame pivotado para el heatmap
    heatmap_data = employment_type_counts.set_index('employment_type').T
    
    # Crear el heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data=heatmap_data, annot=True, fmt="d", cmap='viridis', linewidths=.5)
    plt.title('Distribución de tipos de empleo')
    plt.xlabel('Tipo de empleo')
    plt.ylabel('Conteo')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 10. ¿Cómo se distribuyen los tipos de empleo en diferentes regiones y categorías de trabajo?

def plot_employment_type_by_region_percentage(data):
    # Calcular el porcentaje de cada tipo de empleo en cada región
    region_employment_type_counts = data.groupby('region')['employment_type'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    region_employment_type_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Distribución de tipos de empleo por región (en porcentaje)')
    plt.xlabel('Región')
    plt.ylabel('Porcentaje')
    plt.legend(title='Tipo de empleo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_employment_type_by_job_category_percentage(data):
    # Calcular el porcentaje de cada tipo de empleo en cada categoría de trabajo
    job_category_employment_type_counts = data.groupby('job_category')['employment_type'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    job_category_employment_type_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Distribución de tipos de empleo por categoría de trabajo (en porcentaje)')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Porcentaje')
    plt.legend(title='Tipo de empleo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 11. ¿Qué categorías de trabajo tienen los salarios más altos?

def plot_salary_by_job_category(data):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='job_category', y='salary_in_usd_2023', data=data)
    plt.title('Distribución de salarios por categoría de trabajo')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
# 12. ¿Cómo se comparan las diferentes categorías de trabajo en términos de modalidad y localización?

def plot_work_setting_by_job_category(data):
    # Calcular el porcentaje de cada modalidad de trabajo en cada categoría de trabajo
    job_category_work_setting_counts = data.groupby('job_category')['work_setting'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    job_category_work_setting_counts.plot(kind='bar', stacked=True, figsize=(16, 8))
    plt.title('Distribución de modalidades de trabajo por categoría de trabajo (en porcentaje)')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Porcentaje')
    plt.legend(title='Modalidad de trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_work_setting_by_region_v2(data):
    # Calcular el porcentaje de cada modalidad de trabajo en cada región
    region_work_setting_counts = data.groupby('region')['work_setting'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    region_work_setting_counts.plot(kind='bar', stacked=True, figsize=(16, 8))
    plt.title('Distribución de modalidades de trabajo por región (en porcentaje)')
    plt.xlabel('Región')
    plt.ylabel('Porcentaje')
    plt.legend(title='Modalidad de trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.show()


# 13. ¿Cuáles son los títulos de trabajo más comunes en el dataset?

def plot_common_job_titles(data, top_n=10):
    plt.figure(figsize=(12, 6))
    common_job_titles = data['job_title'].value_counts().head(top_n)
    sns.barplot(x=common_job_titles.values, y=common_job_titles.index)
    plt.title(f'Títulos de trabajo más comunes (Top {top_n})')
    plt.xlabel('Frecuencia')
    plt.ylabel('Título de trabajo')
    plt.tight_layout()
    plt.show()

# 14. ¿Cómo varía el salario entre diferentes títulos de trabajo?

def plot_salary_by_job_title(data, top_n=10):
    plt.figure(figsize=(16, 8))
    common_job_titles = data['job_title'].value_counts().index[:top_n]
    sns.boxplot(x='job_title', y='salary_in_usd_2023', data=data[data['job_title'].isin(common_job_titles)])
    plt.title(f'Distribución de salarios por título de trabajo (Top {top_n})')
    plt.xlabel('Título de trabajo')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(rotation=45)
    #plt.tight_layout()
    plt.show()



# 15. ¿Cómo influye la cantidad de años de experiencia en el salario y la probabilidad de obtener empleo en diferentes categorías y localizaciones?

def plot_experience_vs_salary(data):
    # Convertir 'experience_level' en valores numéricos
    experience_mapping = {'Entry-level': 1, 'Mid-level': 2, 'Senior': 3, 'Executive': 4}
    data['experience_level_numeric'] = data['experience_level'].map(experience_mapping)
    
    # Asegurarse de que 'salary_in_usd_2023' es de tipo float
    data['salary_in_usd_2023'] = data['salary_in_usd_2023'].astype(float)
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='experience_level_numeric', y='salary_in_usd_2023', data=data, alpha=0.5)
    sns.regplot(x='experience_level_numeric', y='salary_in_usd_2023', data=data, scatter=False, color='red')
    plt.title('Relación entre años de experiencia y salario')
    plt.xlabel('Nivel de experiencia (numérico)')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.xticks(ticks=[1, 2, 3, 4], labels=['Entry-level', 'Mid-level', 'Senior', 'Executive'])
    plt.tight_layout()
    plt.show()

def plot_experience_distribution_by_job_category(data):
    # Convertir 'experience_level' en valores numéricos
    experience_mapping = {'Entry-level': 1, 'Mid-level': 2, 'Senior': 3, 'Executive': 4}
    data['experience_level_numeric'] = data['experience_level'].map(experience_mapping)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='job_category', y='experience_level_numeric', data=data, palette='viridis')
    plt.title('Distribución de años de experiencia por categoría de trabajo')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Nivel de experiencia (numérico)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_experience_distribution_by_region(data):
    # Convertir 'experience_level' en valores numéricos
    experience_mapping = {'Entry-level': 1, 'Mid-level': 2, 'Senior': 3, 'Executive': 4}
    data['experience_level_numeric'] = data['experience_level'].map(experience_mapping)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='region', y='experience_level_numeric', data=data, palette='viridis')
    plt.title('Distribución de años de experiencia por región')
    plt.xlabel('Región')
    plt.ylabel('Nivel de experiencia (numérico)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 16. ¿Qué tamaño de empresa tiende a contratar más para cada categoría de trabajo?

def plot_company_size_by_job_category(data):
    # Calcular el porcentaje de cada tamaño de empresa en cada categoría de trabajo
    job_category_company_size_counts = data.groupby('job_category')['company_size'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    job_category_company_size_counts.plot(kind='bar', stacked=True, figsize=(16, 8))
    plt.title('Distribución del tamaño de la empresa por categoría de trabajo (en porcentaje)')
    plt.xlabel('Categoría de trabajo')
    plt.ylabel('Porcentaje')
    plt.legend(title='Tamaño de la empresa', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 17. ¿Hay diferencias en los salarios ofrecidos por pequeñas, medianas y grandes empresas?

def plot_salary_by_company_size(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='company_size', y='salary_in_usd_2023', data=data)
    plt.title('Distribución de salarios por tamaño de empresa')
    plt.xlabel('Tamaño de empresa')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.tight_layout()
    plt.show()
    
# 18. ¿Qué regiones prefieren más el trabajo remoto, híbrido o en persona?

def plot_work_setting_by_region_v3(data):
    # Calcular el porcentaje de cada modalidad de trabajo en cada región
    region_work_setting_counts = data.groupby('region')['work_setting'].value_counts(normalize=True).unstack() * 100
    
    # Crear el gráfico de barras apiladas
    region_work_setting_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Distribución de modalidades de trabajo por región (en porcentaje)')
    plt.xlabel('Región')
    plt.ylabel('Porcentaje')
    plt.legend(title='Modalidad de trabajo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    #plt.tight_layout()
    plt.show()
    
# 19. ¿Hay variaciones salariales significativas según la modalidad de trabajo en diferentes regiones?

def plot_salary_by_work_setting_and_region(data):
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='work_setting', y='salary_in_usd_2023', hue='region', data=data)
    plt.title('Variaciones salariales según modalidad de trabajo y región')
    plt.xlabel('Modalidad de trabajo')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 20. ¿Cuáles son los títulos de trabajo menos comunes en el dataset?
def analyze_least_common_job_titles(df, n=10):
    # Calcular la distribución de 'job_title'
    job_title_counts = df['job_title'].value_counts(ascending=True)

    # Mostrar los títulos de trabajo menos comunes en un DataFrame
    job_title_df = job_title_counts.reset_index()
    job_title_df.columns = ['job_title', 'Frecuencia']
    print(f"Títulos de trabajo menos comunes (top {n}):")
    print(job_title_df.head(n))  # Mostrar los n títulos menos comunes

    # Visualizar los títulos de trabajo menos comunes
    plt.figure(figsize=(12, 6))
    sns.barplot(y=job_title_df['job_title'].head(n), x=job_title_df['Frecuencia'].head(n))
    plt.title('Títulos de Trabajo Menos Comunes')
    plt.xlabel('Frecuencia')
    plt.ylabel('Título de Trabajo')
    plt.show()


# 21. ¿Cuál es la distribución de los salarios en el dataset y cuáles son las ofertas que generan los datos extremos (salarios más altos y más bajos)?
def plot_salary_distribution(data):
    plt.figure(figsize=(14, 8))
    sns.histplot(data['salary_in_usd_2023'], kde=True, color='purple')
    plt.title('Distribución de salarios en USD (ajustado a 2023)')
    plt.xlabel('Salario en USD (ajustado a 2023)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def plot_salary_boxplot(data):
    plt.figure(figsize=(14, 8))
    sns.boxplot(y='salary_in_usd_2023', data=data, color='purple')
    plt.title('Boxplot de salarios en USD (ajustado a 2023)')
    plt.ylabel('Salario en USD (ajustado a 2023)')
    plt.show()

def identify_extreme_salaries(data):
    # Ordenar los salarios de menor a mayor
    sorted_data = data.sort_values(by='salary_in_usd_2023')
    
    # Obtener las 10 ofertas con salarios más bajos
    lowest_salaries = sorted_data.head(10)
    
    # Obtener las 10 ofertas con salarios más altos
    highest_salaries = sorted_data.tail(10)
    
    return lowest_salaries, highest_salaries

def display_tables(lowest_salaries, highest_salaries):
    # Mostrar las 10 ofertas con salarios más bajos
    print("Ofertas con los 10 salarios más bajos:")
    display(lowest_salaries)
    
    # Mostrar las 10 ofertas con salarios más altos
    print("\nOfertas con los 10 salarios más altos:")
    display(highest_salaries)






# 22. ¿Cuáles son las variables con mayor correlación con el salario en el dataset?


def analyze_salary_correlations(data):
    # Eliminar las columnas irrelevantes
    data_encoded = data.drop(columns=['salary_currency', 'salary', 'salary_in_usd', 'inflation_adjustment']).copy()
    
    # Crear una variable categórica ordinal para 'experience_level'
    experience_level_mapping = {
        'Entry-level': 1,
        'Mid-level': 2,
        'Senior': 3,
        'Executive': 4
    }
    data_encoded['experience_level_ordinal'] = data_encoded['experience_level'].map(experience_level_mapping)

    # Crear una variable categórica nominal para 'experience_level'
    data_encoded['experience_level_nominal'] = data_encoded['experience_level'].astype('category').cat.codes

    # Eliminar la columna original 'experience_level'
    data_encoded = data_encoded.drop(columns=['experience_level'])

    # Convertir otras variables categóricas en variables numéricas usando 'astype' y 'cat.codes'
    data_encoded['work_setting'] = data_encoded['work_setting'].astype('category').cat.codes
    data_encoded['company_size'] = data_encoded['company_size'].astype('category').cat.codes
    data_encoded['job_category'] = data_encoded['job_category'].astype('category').cat.codes
    data_encoded['region'] = data_encoded['region'].astype('category').cat.codes
    data_encoded['employment_type'] = data_encoded['employment_type'].astype('category').cat.codes
    data_encoded['job_title'] = data_encoded['job_title'].astype('category').cat.codes
    data_encoded['work_year'] = data_encoded['work_year'].astype('category').cat.codes
    data_encoded['company_location'] = data_encoded['company_location'].astype('category').cat.codes
    data_encoded['employee_residence'] = data_encoded['employee_residence'].astype('category').cat.codes
    data_encoded['salary_in_usd_2023'] = data_encoded['salary_in_usd_2023'].astype(float)

    # Calcular la matriz de correlación
    correlation_matrix = data_encoded.corr()

    # Seleccionar las correlaciones con 'salary_in_usd_2023'
    salary_correlations = correlation_matrix['salary_in_usd_2023'].sort_values(ascending=False)

    # Visualizar las correlaciones más altas
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', lw='.5', cmap='crest')
    plt.title('Matriz de correlación')
    plt.show()

    # Devolver las correlaciones para el análisis
    return salary_correlations


# 23. ¿Cómo podemos predecir el salario utilizando modelos de regresión y cuáles son sus métricas de evaluación?

def train_and_evaluate_models(data):
    # Eliminar las columnas irrelevantes y codificar las variables categóricas
    data_encoded = data.drop(columns=['job_title', 'work_year', 'salary_currency', 'salary', 'salary_in_usd', 'employee_residence', 'company_location', 'inflation_adjustment']).copy()
    data_encoded['experience_level'] = data_encoded['experience_level'].astype('category').cat.codes
    data_encoded['work_setting'] = data_encoded['work_setting'].astype('category').cat.codes
    data_encoded['company_size'] = data_encoded['company_size'].astype('category').cat.codes
    data_encoded['job_category'] = data_encoded['job_category'].astype('category').cat.codes
    data_encoded['region'] = data_encoded['region'].astype('category').cat.codes
    data_encoded['employment_type'] = data_encoded['employment_type'].astype('category').cat.codes
    data_encoded['salary_in_usd_2023'] = data_encoded['salary_in_usd_2023'].astype(float)

    # Definir las características y la variable objetivo
    X = data_encoded.drop(columns=['salary_in_usd_2023'])
    y = data_encoded['salary_in_usd_2023']

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar y evaluar un modelo de regresión lineal
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    lin_rmse = mean_squared_error(y_test, y_pred_lin, squared=False)
    lin_mae = mean_absolute_error(y_test, y_pred_lin)
    lin_r2 = r2_score(y_test, y_pred_lin)

    # Entrenar y evaluar un modelo de Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)

    # Imprimir las métricas de evaluación de ambos modelos
    print("Modelos Comparados:")
    print("Regresión Lineal:")
    print(f"  RMSE: {lin_rmse}")
    print(f"  MAE: {lin_mae}")
    print(f"  R²: {lin_r2}")
    print("Random Forest:")
    print(f"  RMSE: {rf_rmse}")
    print(f"  MAE: {rf_mae}")
    print(f"  R²: {rf_r2}")

    # Comparar los modelos y seleccionar el mejor
    if rf_r2 > lin_r2:
        best_model = rf_model
        best_rmse = rf_rmse
        best_mae = rf_mae
        best_r2 = rf_r2
        y_pred_best = y_pred_rf
        model_name = "Random Forest"
    else:
        best_model = lin_model
        best_rmse = lin_rmse
        best_mae = lin_mae
        best_r2 = lin_r2
        y_pred_best = y_pred_lin
        model_name = "Linear Regression"

    # Imprimir las métricas de evaluación del mejor modelo
    print(f"\nMejor Modelo: {model_name}")
    print(f"RMSE: {best_rmse}")
    print(f"MAE: {best_mae}")
    print(f"R²: {best_r2}")

    # Visualizar las predicciones vs. valores reales para el mejor modelo
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicciones vs. Valores Reales ({model_name})')
    plt.show()

# 3.6 Visualización del Rendimiento del Modelo

def graficar_dispersion(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Valores Predichos vs Valores Reales')
    plt.grid(True)
    plt.show()

def graficar_residuos(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuales')
    plt.title('Distribución de Residuales')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuales')
    plt.title('Residuales vs Valores Predichos')
    plt.grid(True)
    plt.show()

def graficar_importancia_caracteristicas(model, X_train_scaled):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train_scaled.columns

    plt.figure(figsize=(12, 6))
    plt.title("Importancia de las Características")
    plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X_train_scaled.shape[1]])
    plt.show()


