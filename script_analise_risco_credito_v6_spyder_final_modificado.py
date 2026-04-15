#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análise de risco de crédito utilizando o dataset Home Credit Default Risk.
Compara o desempenho de Regressão Logística (sem tuning), LightGBM e XGBoost (com tuning).
Versão 6 Final (Spyder): Nomes em português, Engenharia de Atributos, Threshold 20%.

Autor: TCC Risco de Crédito (Atualizado)
Data: Junho 2025
"""

# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import shap
from tqdm import tqdm
import warnings
import sys # Para checar versão do Python se necessário
import sklearn # Para checar versão do scikit-learn

warnings.filterwarnings("ignore")

# Configuração para exibição de gráficos no Spyder e alta resolução
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 150
plt.style.use("ggplot")

# Definição de funções auxiliares

def carregar_dados(caminho_arquivo):
    """Carrega os dados do arquivo CSV."""
    print(f"Carregando dados de {caminho_arquivo}...")
    try:
        return pd.read_csv(caminho_arquivo)
    except FileNotFoundError:
        print(f"ERRO: Arquivo {caminho_arquivo} não encontrado!")
        print("Por favor, baixe o dataset \"Home Credit Default Risk\" do Kaggle e coloque \"application_train.csv\" no mesmo diretório do script.")
        sys.exit(1) # Termina o script se o arquivo não for encontrado
    except Exception as e:
        print(f"ERRO ao carregar o arquivo CSV: {e}")
        sys.exit(1)

def traduzir_renomear_colunas(df):
    """Traduz e renomeia as colunas do DataFrame para português minúsculo."""
    print("Traduzindo e renomeando colunas...")
    colunas_traduzidas = {
        # Dicionário completo de tradução (essencial estar completo)
        # Base
        'SK_ID_CURR': 'id_cliente', 'TARGET': 'alvo', 'NAME_CONTRACT_TYPE': 'tipo_contrato',
        'CODE_GENDER': 'genero', 'FLAG_OWN_CAR': 'possui_carro', 'FLAG_OWN_REALTY': 'possui_imovel',
        'CNT_CHILDREN': 'qtd_filhos', 'AMT_INCOME_TOTAL': 'renda_total', 'AMT_CREDIT': 'valor_credito',
        'AMT_ANNUITY': 'valor_anuidade', 'AMT_GOODS_PRICE': 'valor_bens', 'NAME_TYPE_SUITE': 'acomodacao',
        'NAME_INCOME_TYPE': 'tipo_renda', 'NAME_EDUCATION_TYPE': 'escolaridade', 'NAME_FAMILY_STATUS': 'estado_civil',
        'NAME_HOUSING_TYPE': 'tipo_moradia', 'REGION_POPULATION_RELATIVE': 'populacao_regiao',
        'DAYS_BIRTH': 'dias_nascimento', 'DAYS_EMPLOYED': 'dias_empregado', 'DAYS_REGISTRATION': 'dias_registro',
        'DAYS_ID_PUBLISH': 'dias_emissao_id', 'OWN_CAR_AGE': 'idade_carro', 'FLAG_MOBIL': 'possui_celular',
        'FLAG_EMP_PHONE': 'possui_tel_emprego', 'FLAG_WORK_PHONE': 'possui_tel_trabalho',
        'FLAG_CONT_MOBILE': 'possui_celular_contato', 'FLAG_PHONE': 'possui_telefone', 'FLAG_EMAIL': 'possui_email',
        'OCCUPATION_TYPE': 'profissao', 'CNT_FAM_MEMBERS': 'qtd_membros_familia',
        'REGION_RATING_CLIENT': 'avaliacao_regiao_cliente', 'REGION_RATING_CLIENT_W_CITY': 'avaliacao_regiao_cidade_cliente',
        'WEEKDAY_APPR_PROCESS_START': 'dia_semana_aprovacao', 'HOUR_APPR_PROCESS_START': 'hora_aprovacao',
        'REG_REGION_NOT_LIVE_REGION': 'regiao_registro_diferente_residencia',
        'REG_REGION_NOT_WORK_REGION': 'regiao_registro_diferente_trabalho',
        'LIVE_REGION_NOT_WORK_REGION': 'regiao_residencia_diferente_trabalho',
        'REG_CITY_NOT_LIVE_CITY': 'cidade_registro_diferente_residencia',
        'REG_CITY_NOT_WORK_CITY': 'cidade_registro_diferente_trabalho',
        'LIVE_CITY_NOT_WORK_CITY': 'cidade_residencia_diferente_trabalho',
        'ORGANIZATION_TYPE': 'tipo_organizacao', 'EXT_SOURCE_1': 'fonte_ext_1', 'EXT_SOURCE_2': 'fonte_ext_2',
        'EXT_SOURCE_3': 'fonte_ext_3',
        # Colunas de informações sobre o imóvel (AVG, MODE, MEDI)
        'APARTMENTS_AVG': 'apartamentos_avg', 'BASEMENTAREA_AVG': 'area_porao_avg', 'YEARS_BEGINEXPLUATATION_AVG': 'anos_inicio_uso_avg',
        'YEARS_BUILD_AVG': 'anos_construcao_avg', 'COMMONAREA_AVG': 'area_comum_avg', 'ELEVATORS_AVG': 'elevadores_avg',
        'ENTRANCES_AVG': 'entradas_avg', 'FLOORSMAX_AVG': 'andares_max_avg', 'FLOORSMIN_AVG': 'andares_min_avg',
        'LANDAREA_AVG': 'area_terreno_avg', 'LIVINGAPARTMENTS_AVG': 'aptos_residenciais_avg', 'LIVINGAREA_AVG': 'area_residencial_avg',
        'NONLIVINGAPARTMENTS_AVG': 'aptos_nao_residenciais_avg', 'NONLIVINGAREA_AVG': 'area_nao_residencial_avg',
        'APARTMENTS_MODE': 'apartamentos_mode', 'BASEMENTAREA_MODE': 'area_porao_mode', 'YEARS_BEGINEXPLUATATION_MODE': 'anos_inicio_uso_mode',
        'YEARS_BUILD_MODE': 'anos_construcao_mode', 'COMMONAREA_MODE': 'area_comum_mode', 'ELEVATORS_MODE': 'elevadores_mode',
        'ENTRANCES_MODE': 'entradas_mode', 'FLOORSMAX_MODE': 'andares_max_mode', 'FLOORSMIN_MODE': 'andares_min_mode',
        'LANDAREA_MODE': 'area_terreno_mode', 'LIVINGAPARTMENTS_MODE': 'aptos_residenciais_mode', 'LIVINGAREA_MODE': 'area_residencial_mode',
        'NONLIVINGAPARTMENTS_MODE': 'aptos_nao_residenciais_mode', 'NONLIVINGAREA_MODE': 'area_nao_residencial_mode',
        'APARTMENTS_MEDI': 'apartamentos_medi', 'BASEMENTAREA_MEDI': 'area_porao_medi', 'YEARS_BEGINEXPLUATATION_MEDI': 'anos_inicio_uso_medi',
        'YEARS_BUILD_MEDI': 'anos_construcao_medi', 'COMMONAREA_MEDI': 'area_comum_medi', 'ELEVATORS_MEDI': 'elevadores_medi',
        'ENTRANCES_MEDI': 'entradas_medi', 'FLOORSMAX_MEDI': 'andares_max_medi', 'FLOORSMIN_MEDI': 'andares_min_medi',
        'LANDAREA_MEDI': 'area_terreno_medi', 'LIVINGAPARTMENTS_MEDI': 'aptos_residenciais_medi', 'LIVINGAREA_MEDI': 'area_residencial_medi',
        'NONLIVINGAPARTMENTS_MEDI': 'aptos_nao_residenciais_medi', 'NONLIVINGAREA_MEDI': 'area_nao_residencial_medi',
        'FONDKAPREMONT_MODE': 'fundo_reparo_predial', 'HOUSETYPE_MODE': 'tipo_casa', 'TOTALAREA_MODE': 'area_total',
        'WALLSMATERIAL_MODE': 'material_parede', 'EMERGENCYSTATE_MODE': 'estado_emergencia',
        # Círculo social
        'OBS_30_CNT_SOCIAL_CIRCLE': 'obs_30_circulo_social', 'DEF_30_CNT_SOCIAL_CIRCLE': 'inadimpl_30_circulo_social',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'obs_60_circulo_social', 'DEF_60_CNT_SOCIAL_CIRCLE': 'inadimpl_60_circulo_social',
        'DAYS_LAST_PHONE_CHANGE': 'dias_ultima_troca_tel',
        # Documentos
        'FLAG_DOCUMENT_2': 'flag_documento_2', 'FLAG_DOCUMENT_3': 'flag_documento_3', 'FLAG_DOCUMENT_4': 'flag_documento_4',
        'FLAG_DOCUMENT_5': 'flag_documento_5', 'FLAG_DOCUMENT_6': 'flag_documento_6', 'FLAG_DOCUMENT_7': 'flag_documento_7',
        'FLAG_DOCUMENT_8': 'flag_documento_8', 'FLAG_DOCUMENT_9': 'flag_documento_9', 'FLAG_DOCUMENT_10': 'flag_documento_10',
        'FLAG_DOCUMENT_11': 'flag_documento_11', 'FLAG_DOCUMENT_12': 'flag_documento_12', 'FLAG_DOCUMENT_13': 'flag_documento_13',
        'FLAG_DOCUMENT_14': 'flag_documento_14', 'FLAG_DOCUMENT_15': 'flag_documento_15', 'FLAG_DOCUMENT_16': 'flag_documento_16',
        'FLAG_DOCUMENT_17': 'flag_documento_17', 'FLAG_DOCUMENT_18': 'flag_documento_18', 'FLAG_DOCUMENT_19': 'flag_documento_19',
        'FLAG_DOCUMENT_20': 'flag_documento_20', 'FLAG_DOCUMENT_21': 'flag_documento_21',
        # Consultas ao bureau
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'consultas_credito_hora', 'AMT_REQ_CREDIT_BUREAU_DAY': 'consultas_credito_dia',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'consultas_credito_semana', 'AMT_REQ_CREDIT_BUREAU_MON': 'consultas_credito_mes',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'consultas_credito_trimestre', 'AMT_REQ_CREDIT_BUREAU_YEAR': 'consultas_credito_ano'
    }

    # Renomear colunas existentes no DataFrame
    df.rename(columns=colunas_traduzidas, inplace=True)
    # Garantir que todos os nomes estejam em minúsculo (caso alguma chave do dict esteja errada)
    df.columns = df.columns.str.lower()
    return df

def engenharia_atributos(df):
    """Cria novas features baseadas nas existentes."""
    print("Aplicando engenharia de atributos...")
    # Idade em anos (corrigindo potencial erro se 'dias_nascimento' não existir)
    if 'dias_nascimento' in df.columns:
        df['idade_anos'] = (-df['dias_nascimento']) // 365
    else:
        print("Aviso: Coluna 'dias_nascimento' não encontrada para calcular 'idade_anos'.")

    # Ratios de crédito (verificando existência das colunas)
    required_cols = ['valor_credito', 'renda_total', 'valor_anuidade', 'valor_bens']
    if all(col in df.columns for col in required_cols):
        df['credito_por_renda'] = df['valor_credito'] / (df['renda_total'] + 1e-6) # Evitar divisão por zero
        df['anuidade_por_renda'] = df['valor_anuidade'] / (df['renda_total'] + 1e-6)
        df['credito_por_anuidade'] = df['valor_credito'] / (df['valor_anuidade'] + 1e-6)
        df['credito_por_valor_bens'] = df['valor_credito'] / (df['valor_bens'] + 1e-6)
        # Tratar valores infinitos que podem surgir
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print("Aviso: Colunas necessárias para ratios de crédito não encontradas.")

    # Corrigir valor anômalo em 'dias_empregado'
    if 'dias_empregado' in df.columns:
        df['dias_empregado'].replace(365243, np.nan, inplace=True)

    return df

def analisar_dados_faltantes(df):
    """Analisa e exibe informações sobre dados faltantes."""
    missing_values = df.isnull().sum()
    missing_values_percent = 100 * missing_values / len(df)
    missing_info = pd.DataFrame({
        "Valores Faltantes": missing_values,
        "Porcentagem (%)": missing_values_percent
    })
    missing_info = missing_info[missing_info["Valores Faltantes"] > 0].sort_values(
        "Porcentagem (%)", ascending=False
    )
    return missing_info

def remover_colunas_com_muitos_faltantes(df, threshold=20):
    """Remove colunas com porcentagem de valores faltantes acima do threshold."""
    missing_percent = df.isnull().mean() * 100
    colunas_para_remover = missing_percent[missing_percent > threshold].index.tolist()
    print(f"Removendo {len(colunas_para_remover)} colunas com mais de {threshold}% de valores faltantes:")
    if len(colunas_para_remover) > 0:
        print(colunas_para_remover)
    # Garantir que a coluna 'alvo' não seja removida acidentalmente
    if 'alvo' in colunas_para_remover:
        print("Aviso: Tentativa de remover a coluna 'alvo'. Mantendo a coluna alvo.")
        colunas_para_remover.remove('alvo')
    return df.drop(columns=colunas_para_remover)

def analisar_distribuicao_target(df, target_col="alvo"):
    """Analisa e exibe a distribuição da variável target."""
    if target_col not in df.columns:
        print(f"ERRO: Coluna alvo '{target_col}' não encontrada no DataFrame.")
        return
    target_counts = df[target_col].value_counts()
    target_percent = 100 * target_counts / len(df)
    print("\nDistribuição da variável alvo:")
    print(f"0 (Não inadimplente): {target_counts.get(0, 0)} ({target_percent.get(0, 0):.2f}%)")
    print(f"1 (Inadimplente): {target_counts.get(1, 0)} ({target_percent.get(1, 0):.2f}%)")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title("Distribuição da Variável Alvo")
    plt.xlabel("Alvo (0 = Não Inadimplente, 1 = Inadimplente)")
    plt.ylabel("Contagem")
    plt.savefig("distribuicao_alvo_spyder_final.png", dpi=300)
    plt.close()

def calcular_indice_gini(auc_score):
    """Calcula o índice de Gini a partir do AUC-ROC."""
    return 2 * auc_score - 1

def plotar_matriz_confusao(y_true, y_pred, titulo, nome_arquivo):
    """Plota e salva a matriz de confusão com alta resolução."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(titulo)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()

def plotar_curva_roc(y_true, y_proba, modelos, nome_arquivo):
    """Plota e salva a curva ROC para múltiplos modelos com alta resolução."""
    plt.figure(figsize=(10, 8))
    for i, (proba, modelo) in enumerate(zip(y_proba, modelos)):
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        gini = calcular_indice_gini(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f"{modelo} (AUC = {roc_auc:.3f}, Gini = {gini:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC Comparativa")
    plt.legend(loc="lower right")
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()

def get_feature_names_from_transformer(column_transformer):
    """Obtém os nomes das features após a transformação do ColumnTransformer.
       Versão mais robusta usando get_feature_names_out quando disponível.
    """
    output_features = []
    try:
        # Scikit-learn >= 0.24 (aproximadamente)
        if hasattr(column_transformer, 'get_feature_names_out'):
            output_features = list(column_transformer.get_feature_names_out())
        else:
            # Versões mais antigas ou fallback
            for name, pipe, features in column_transformer.transformers_:
                if name == 'remainder':
                    # Tenta obter nomes do remainder se não for 'drop'
                    if pipe != 'drop' and hasattr(column_transformer, '_feature_names_in'):
                        remainder_cols = [col for col in column_transformer._feature_names_in
                                        if col not in column_transformer.transformers_[0][2] + column_transformer.transformers_[1][2]]
                        output_features.extend(remainder_cols)
                    continue
                if pipe == 'drop':
                    continue

                if hasattr(pipe, 'steps'): # Pipeline
                    last_step = pipe.steps[-1][1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        # Prefixo automático do pipeline: name + '__' + step_name + '__'
                        # Prefixo do OneHotEncoder: feature_name + '_'
                        # Precisamos construir o nome completo
                        step_feature_names = last_step.get_feature_names_out(features)
                        output_features.extend(step_feature_names)
                    elif hasattr(last_step, 'get_feature_names'): # Older versions
                        step_feature_names = last_step.get_feature_names(features)
                        output_features.extend(step_feature_names)
                    else:
                        output_features.extend(features) # Fallback
                else: # Direct transformer
                    if hasattr(pipe, 'get_feature_names_out'):
                        output_features.extend(pipe.get_feature_names_out(features))
                    elif hasattr(pipe, 'get_feature_names'): # Older versions
                        output_features.extend(pipe.get_feature_names(features))
                    else:
                        output_features.extend(features) # Fallback
    except Exception as e:
        print(f"AVISO: Erro ao obter nomes das features do ColumnTransformer: {e}. Usando nomes genéricos.")
        # Fallback muito genérico se tudo falhar
        try:
            num_output_features = column_transformer.transform(pd.DataFrame(columns=column_transformer.feature_names_in_)).shape[1]
            output_features = [f"feature_{i}" for i in range(num_output_features)]
        except:
             output_features = [] # Retorna vazio se não conseguir nem estimar

    return output_features

def analisar_shap_values(modelo_final, X_test_transformado, feature_names, nome_modelo, nome_arquivo_base):
    """Analisa e visualiza os valores SHAP para interpretabilidade."""
    print(f"\nAnalisando SHAP values para {nome_modelo}...")
    try:
        if not feature_names:
             print("ERRO: Lista de nomes de features está vazia para análise SHAP. Abortando SHAP.")
             return pd.DataFrame()

        # Garantir que X_test_transformado seja um DataFrame para SHAP
        if not isinstance(X_test_transformado, pd.DataFrame):
            if X_test_transformado.shape[1] == len(feature_names):
                X_test_shap_df = pd.DataFrame(X_test_transformado, columns=feature_names)
            else:
                print(f"ERRO: Discrepância no número de colunas ({X_test_transformado.shape[1]}) vs nomes ({len(feature_names)}) e X não é DataFrame. Abortando SHAP.")
                return pd.DataFrame()
        else:
            X_test_shap_df = X_test_transformado # Já é DataFrame
            # Verificar se as colunas do DataFrame correspondem aos nomes fornecidos
            if list(X_test_shap_df.columns) != feature_names:
                 print("AVISO: Nomes de features fornecidos diferem das colunas do DataFrame X_test_transformado. Usando colunas do DataFrame.")
                 feature_names = list(X_test_shap_df.columns)

        # Selecionar o explainer apropriado
        if isinstance(modelo_final, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(modelo_final)
            shap_values = explainer.shap_values(X_test_shap_df)
            # Para classificação binária, TreeExplainer retorna lista [shap_class0, shap_class1]
            shap_values_positive = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
        elif isinstance(modelo_final, LogisticRegression):
            # LinearExplainer espera máscara ou dados de referência
            explainer = shap.LinearExplainer(modelo_final, X_test_shap_df)
            shap_values = explainer.shap_values(X_test_shap_df)
            shap_values_positive = shap_values # Para LinearExplainer, geralmente é direto
        else:
            print(f"AVISO: Tentando shap.KernelExplainer genérico para {type(modelo_final)}. Isso pode ser lento.")
            # KernelExplainer precisa de uma amostra de background (ex: do treino)
            # Usar X_test_shap_df como background é uma aproximação
            explainer = shap.KernelExplainer(modelo_final.predict_proba, X_test_shap_df)
            shap_values = explainer.shap_values(X_test_shap_df) # Calcula para ambas as classes
            shap_values_positive = shap_values[1] # Seleciona SHAP para a classe positiva (1)

        # Plotando summary plot
        plt.figure()
        shap.summary_plot(shap_values_positive, X_test_shap_df, show=False, feature_names=feature_names)
        plt.title(f"SHAP Summary Plot - {nome_modelo}")
        plt.tight_layout()
        plt.savefig(f"{nome_arquivo_base}_summary_spyder_final.png", dpi=300)
        plt.close()

        # Plotando bar plot
        plt.figure()
        shap.summary_plot(shap_values_positive, X_test_shap_df, plot_type="bar", show=False, feature_names=feature_names)
        plt.title(f"SHAP Feature Importance - {nome_modelo}")
        plt.tight_layout()
        plt.savefig(f"{nome_arquivo_base}_importance_spyder_final.png", dpi=300)
        plt.close()

        # Retornar importância média absoluta
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance_SHAP_Abs_Mean": np.abs(shap_values_positive).mean(axis=0)
        }).sort_values("Importance_SHAP_Abs_Mean", ascending=False)

    except Exception as e:
        print(f"ERRO ao calcular/plotar SHAP para {nome_modelo}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """Função principal que executa todo o pipeline de análise."""
    print("Iniciando análise de risco de crédito (v6 Final - Spyder, PT-BR, Eng. Atrib., Thr 20%)...")
    print(f"Versão Scikit-learn: {sklearn.__version__}")

    # Carregamento dos dados
    caminho_arquivo = "application_train.csv"
    df_original = carregar_dados(caminho_arquivo)
    print(f"Dados originais carregados. Shape: {df_original.shape}")

    # Traduzir e Renomear Colunas
    df_renamed = traduzir_renomear_colunas(df_original.copy())

    # Engenharia de Atributos
    df_featured = engenharia_atributos(df_renamed)

    # Análise inicial e pré-processamento
    print("\nAnalisando dados faltantes após engenharia de atributos...")
    missing_info = analisar_dados_faltantes(df_featured)
    print(f"Colunas com valores faltantes: {len(missing_info)}")

    # Remover colunas com mais de 20% de valores faltantes
    df_cleaned = remover_colunas_com_muitos_faltantes(df_featured, threshold=20)
    print(f"\nShape após remoção de colunas: {df_cleaned.shape}")

    # Analisar distribuição do alvo
    analisar_distribuicao_target(df_cleaned, target_col="alvo")

    # Separar features (X) e target (y)
    if "alvo" not in df_cleaned.columns:
        print("ERRO: Coluna 'alvo' não encontrada após limpeza. Verifique o processo.")
        sys.exit(1)

    colunas_para_dropar = ["alvo"]
    if "id_cliente" in df_cleaned.columns:
        colunas_para_dropar.append("id_cliente")

    X = df_cleaned.drop(columns=colunas_para_dropar)
    y = df_cleaned["alvo"]

    # Identificar colunas numéricas e categóricas *após* limpeza e engenharia
    colunas_numericas = X.select_dtypes(include=np.number).columns.tolist()
    colunas_categoricas = X.select_dtypes(include="object").columns.tolist()

    print(f"\nNúmero de features numéricas final: {len(colunas_numericas)}")
    print(f"Número de features categóricas final: {len(colunas_categoricas)}")

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nConjunto de treino: {X_train.shape}")
    print(f"Conjunto de teste: {X_test.shape}")

    # Criação do preprocessador
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, colunas_numericas),
            ("cat", categorical_transformer, colunas_categoricas)
        ],
        remainder="passthrough" # Manter colunas não especificadas (se houver)
    )

    # --- Treinamento dos Modelos ---

    # 1. Regressão Logística (SEM TUNING)
    print("\nCriando e treinando pipeline da Regressão Logística (sem tuning)...")
    pipe_lr = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    try:
        pipe_lr.fit(X_train, y_train)
        best_lr = pipe_lr
        print("Regressão Logística treinada com parâmetros padrão.")
    except Exception as e:
        print(f"ERRO ao treinar Regressão Logística: {e}")
        best_lr = None # Marca como falha

    # 2. LightGBM e XGBoost (COM TUNING)
    pipe_lgbm = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", lgb.LGBMClassifier(random_state=42, class_weight="balanced"))
    ])

    # Calcular scale_pos_weight para XGBoost (importante para dados desbalanceados)
    scale_pos_weight_xgb = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    pipe_xgb = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", xgb.XGBClassifier(random_state=42, objective="binary:logistic", scale_pos_weight=scale_pos_weight_xgb, eval_metric="logloss", use_label_encoder=False))
    ])

    # Grades de parâmetros reduzidas para teste mais rápido (ajustar conforme necessário)
    param_grid_lgbm = {
        "modelo__n_estimators": [100, 200],
        # "modelo__learning_rate": [0.05, 0.1],
        "modelo__num_leaves": [31, 50],
    }

    param_grid_xgb = {
        "modelo__n_estimators": [100, 200],
        # "modelo__learning_rate": [0.05, 0.1],
        "modelo__max_depth": [3, 5],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    modelos_para_tuning = {
        "LightGBM": (pipe_lgbm, param_grid_lgbm),
        "XGBoost": (pipe_xgb, param_grid_xgb)
    }

    melhores_modelos_tuned = {}
    best_lgbm = None
    best_xgb = None

    print("\nIniciando Tuning com GridSearchCV (n_jobs=1) para LGBM e XGB...")

    for nome_modelo, (pipeline, params) in tqdm(modelos_para_tuning.items(), desc="Tuning LGBM/XGB", unit="modelo"):
        print(f"\nTuning {nome_modelo}...")
        try:
            grid = GridSearchCV(pipeline, params, cv=cv, scoring="roc_auc", n_jobs=1, verbose=1)
            grid.fit(X_train, y_train)
            melhores_modelos_tuned[nome_modelo] = grid.best_estimator_
            print(f"Melhores parâmetros {nome_modelo}: {grid.best_params_}")
            print(f"Melhor AUC-ROC (CV) {nome_modelo}: {grid.best_score_:.4f}")
            if nome_modelo == "LightGBM":
                best_lgbm = grid.best_estimator_
            elif nome_modelo == "XGBoost":
                best_xgb = grid.best_estimator_
        except Exception as e:
            print(f"ERRO durante o tuning de {nome_modelo}: {e}")
            # Continua para o próximo modelo se um falhar

    # --- Avaliação dos Modelos no Conjunto de Teste ---
    print("\nAvaliando modelos no conjunto de teste...")

    resultados = {}
    modelos_avaliar = {
        "Regressão Logística": best_lr,
        "LightGBM": best_lgbm,
        "XGBoost": best_xgb
    }

    probas_plot = []
    nomes_plot = []

    for nome, modelo in modelos_avaliar.items():
        if modelo is None:
            print(f"Modelo {nome} não foi treinado com sucesso. Pulando avaliação.")
            continue
        try:
            y_proba = modelo.predict_proba(X_test)[:, 1]
            y_pred = modelo.predict(X_test) # Usar predict para matriz de confusão/report
            auc_score = roc_auc_score(y_test, y_proba)
            gini_score = calcular_indice_gini(auc_score)
            resultados[nome] = {"AUC": auc_score, "Gini": gini_score}

            print(f"\n{nome} - AUC-ROC: {auc_score:.4f}, Índice de Gini: {gini_score:.4f}")
            print(f"Relatório de Classificação - {nome}:")
            print(classification_report(y_test, y_pred))

            plotar_matriz_confusao(
                y_test, y_pred,
                f"Matriz de Confusão - {nome}",
                f"matriz_confusao_{nome.lower().replace(' ', '_').replace('(', '').replace(')', '')}_spyder_final.png"
            )
            probas_plot.append(y_proba)
            nomes_plot.append(nome)
        except Exception as e:
            print(f"ERRO ao avaliar {nome}: {e}")

    # Plotar curva ROC comparativa apenas com modelos avaliados com sucesso
    if probas_plot:
        plotar_curva_roc(
            y_test,
            probas_plot,
            nomes_plot,
            "curva_roc_comparacao_spyder_final.png"
        )

    # --- Análise SHAP para Interpretabilidade ---
    print("\nRealizando análise SHAP para interpretabilidade (pode levar tempo)...")

    # Usar um modelo que treinou com sucesso para obter o preprocessador
    modelo_base_shap = best_lr if best_lr else (best_lgbm if best_lgbm else best_xgb)
    if modelo_base_shap:
        try:
            preprocessador_ajustado = modelo_base_shap.named_steps["preprocessador"]
            # Tentar obter nomes das features de forma robusta
            feature_names_final = get_feature_names_from_transformer(preprocessador_ajustado)

            if not feature_names_final:
                 print("ERRO: Não foi possível obter nomes das features para SHAP. Abortando.")
            else:
                # Pré-processar o conjunto de teste (ou amostra) para SHAP
                # Usar uma amostra menor para acelerar, se necessário
                sample_size_shap = min(1000, X_test.shape[0]) # Amostra menor para SHAP
                X_test_sample_shap = X_test.sample(sample_size_shap, random_state=42)
                X_test_sample_prep_shap = preprocessador_ajustado.transform(X_test_sample_shap)
                X_test_sample_prep_shap_df = pd.DataFrame(X_test_sample_prep_shap, columns=feature_names_final)

                # Análise SHAP para modelos treinados com sucesso
                shap_importances = {}
                if best_lgbm:
                    lgbm_model_final = best_lgbm.named_steps["modelo"]
                    shap_importances["LightGBM"] = analisar_shap_values(
                        lgbm_model_final, X_test_sample_prep_shap_df, feature_names_final, "LightGBM", "shap_lgbm"
                    )
                if best_xgb:
                    xgb_model_final = best_xgb.named_steps["modelo"]
                    shap_importances["XGBoost"] = analisar_shap_values(
                        xgb_model_final, X_test_sample_prep_shap_df, feature_names_final, "XGBoost", "shap_xgb"
                    )
                # SHAP para Regressão Logística (opcional, pode ser menos informativo)
                # if best_lr:
                #     lr_model_final = best_lr.named_steps["modelo"]
                #     shap_importances["Regressão Logística"] = analisar_shap_values(
                #         lr_model_final, X_test_sample_prep_shap_df, feature_names_final, "Regressão Logística", "shap_lr"
                #     )

                # Exibir importâncias SHAP
                for nome_modelo, importancia_df in shap_importances.items():
                    if not importancia_df.empty:
                        print(f"\nTop 20 features mais importantes ({nome_modelo} - SHAP):")
                        print(importancia_df.head(20))

        except Exception as e:
            print(f"ERRO durante a preparação ou execução da análise SHAP: {e}")
            import traceback
            traceback.print_exc()
            print("A análise SHAP pode não ter sido concluída ou pode estar incompleta.")
    else:
        print("Nenhum modelo treinou com sucesso para realizar análise SHAP.")

    print("\nAnálise (v6 Final - Spyder) concluída!")
    print("Os resultados e gráficos foram salvos no diretório atual com sufixo \"_spyder_final\".")

# Bloco de execução principal - ESSENCIAL para rodar o script
if __name__ == "__main__":
    main()

