import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import io
import warnings
warnings.filterwarnings('ignore')

def get_port():
    return int(os.environ.get("PORT", 8501))

st.set_page_config(
    page_title="Dashboard Avançado de Vendas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"

# Configuração da página
st.set_page_config(
    page_title="Dashboard Avançado de Vendas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        margin: 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #2e7d32;
        background-color: #f1f8e9;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Carrega dados do arquivo Excel"""
    return pd.read_excel(file)

def auto_detect_columns(df):
    """Detecta automaticamente as colunas baseado nos nomes"""
    column_mapping = {}
    
    # Mapeamento de possíveis nomes de colunas
    date_keywords = ['data', 'date', 'dia', 'mes', 'ano', 'periodo', 'timestamp']
    value_keywords = ['valor', 'preco', 'total', 'vendas', 'revenue', 'price', 'receita']
    product_keywords = ['produto', 'product', 'item', 'nome', 'sku', 'codigo']
    seller_keywords = ['vendedor', 'seller', 'representante', 'agent', 'consultor', 'comercial']
    channel_keywords = ['canal', 'channel', 'origem', 'plataforma', 'meio']
    region_keywords = ['regiao', 'region', 'estado', 'cidade', 'pais', 'country', 'uf']
    customer_keywords = ['cliente', 'customer', 'conta', 'account', 'comprador']
    cost_keywords = ['custo', 'cost', 'despesa', 'gasto', 'expense']
    quantity_keywords = ['quantidade', 'qty', 'volume', 'unidade', 'peca']
    category_keywords = ['categoria', 'category', 'tipo', 'class', 'linha']
    
    def find_column(keywords):
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None
    
    # Detectar colunas
    column_mapping['data'] = find_column(date_keywords)
    column_mapping['valor'] = find_column(value_keywords)
    column_mapping['produto'] = find_column(product_keywords)
    column_mapping['vendedor'] = find_column(seller_keywords)
    column_mapping['canal'] = find_column(channel_keywords)
    column_mapping['regiao'] = find_column(region_keywords)
    column_mapping['cliente'] = find_column(customer_keywords)
    column_mapping['custo'] = find_column(cost_keywords)
    column_mapping['quantidade'] = find_column(quantity_keywords)
    column_mapping['categoria'] = find_column(category_keywords)
    
    return {k: v for k, v in column_mapping.items() if v is not None}

def process_data(df, column_mapping):
    """Processa e limpa os dados"""
    df_processed = df.copy()
    
    # Renomear colunas
    rename_dict = {v: k for k, v in column_mapping.items()}
    df_processed = df_processed.rename(columns=rename_dict)
    
    # Processar dados
    if 'data' in df_processed.columns:
        df_processed['data'] = pd.to_datetime(df_processed['data'], errors='coerce')
        df_processed = df_processed.dropna(subset=['data'])
        df_processed['ano'] = df_processed['data'].dt.year
        df_processed['mes'] = df_processed['data'].dt.month
        df_processed['mes_nome'] = df_processed['data'].dt.strftime('%B')
        df_processed['dia_semana'] = df_processed['data'].dt.day_name()
        df_processed['trimestre'] = df_processed['data'].dt.quarter
    
    if 'valor' in df_processed.columns:
        df_processed['valor'] = pd.to_numeric(df_processed['valor'], errors='coerce')
    
    if 'custo' in df_processed.columns:
        df_processed['custo'] = pd.to_numeric(df_processed['custo'], errors='coerce')
        if 'valor' in df_processed.columns:
            df_processed['margem_lucro'] = df_processed['valor'] - df_processed['custo']
            df_processed['margem_lucro_pct'] = (df_processed['margem_lucro'] / df_processed['valor']) * 100
    
    if 'quantidade' in df_processed.columns:
        df_processed['quantidade'] = pd.to_numeric(df_processed['quantidade'], errors='coerce')
        if 'valor' in df_processed.columns:
            df_processed['preco_unitario'] = df_processed['valor'] / df_processed['quantidade']
    
    return df_processed.dropna()

def create_kpi_dashboard(df):
    """Cria dashboard de KPIs principais"""
    st.markdown('<div class="section-header">📊 Indicadores Principais</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_vendas = df['valor'].sum() if 'valor' in df.columns else 0
        st.metric("💰 Total Vendas", f"R$ {total_vendas:,.0f}")
    
    with col2:
        qtd_vendas = len(df)
        st.metric("📈 Qtd Vendas", f"{qtd_vendas:,}")
    
    with col3:
        ticket_medio = df['valor'].mean() if 'valor' in df.columns and len(df) > 0 else 0
        st.metric("🎯 Ticket Médio", f"R$ {ticket_medio:,.0f}")
    
    with col4:
        margem_media = df['margem_lucro_pct'].mean() if 'margem_lucro_pct' in df.columns else 0
        st.metric("💹 Margem Média", f"{margem_media:.1f}%")
    
    with col5:
        clientes_unicos = df['cliente'].nunique() if 'cliente' in df.columns else 0
        st.metric("👥 Clientes Únicos", f"{clientes_unicos:,}")

def analise_tendencias_temporais(df):
    """Análise de Tendências Temporais"""
    st.markdown('<div class="section-header">📈 Análise de Tendências Temporais</div>', unsafe_allow_html=True)
    
    if 'data' not in df.columns or 'valor' not in df.columns:
        st.warning("Dados de data ou valor não encontrados para análise temporal.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendas mensais com tendência
        monthly_sales = df.groupby(df['data'].dt.to_period('M'))['valor'].sum().reset_index()
        monthly_sales['data'] = monthly_sales['data'].astype(str)
        
        fig = px.line(monthly_sales, x='data', y='valor', 
                     title='Tendência de Vendas Mensais',
                     labels={'valor': 'Vendas (R$)', 'data': 'Mês'})
        
        # Adicionar linha de tendência
        x_numeric = np.arange(len(monthly_sales))
        z = np.polyfit(x_numeric, monthly_sales['valor'], 1)
        p = np.poly1d(z)
        fig.add_scatter(x=monthly_sales['data'], y=p(x_numeric), 
                       mode='lines', name='Tendência', line=dict(dash='dash'))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Análise sazonal por mês
        seasonal_analysis = df.groupby('mes_nome')['valor'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        fig = px.bar(x=seasonal_analysis.index, y=seasonal_analysis.values,
                    title='Padrão Sazonal - Vendas Médias por Mês',
                    labels={'x': 'Mês', 'y': 'Vendas Médias (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise por dia da semana
    col3, col4 = st.columns(2)
    
    with col3:
        weekday_sales = df.groupby('dia_semana')['valor'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig = px.bar(x=weekday_sales.index, y=weekday_sales.values,
                    title='Vendas por Dia da Semana',
                    labels={'x': 'Dia da Semana', 'y': 'Vendas (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Análise trimestral
        quarterly_sales = df.groupby('trimestre')['valor'].sum()
        
        fig = px.pie(values=quarterly_sales.values, 
                    names=[f'Q{i}' for i in quarterly_sales.index],
                    title='Distribuição de Vendas por Trimestre')
        st.plotly_chart(fig, use_container_width=True)

def analise_desempenho_produto(df):
    """Análise de Desempenho por Produto"""
    st.markdown('<div class="section-header">🛍️ Análise de Desempenho por Produto</div>', unsafe_allow_html=True)
    
    if 'produto' not in df.columns:
        st.warning("Dados de produto não encontrados.")
        return
    
    # Métricas por produto
    product_metrics = df.groupby('produto').agg({
        'valor': ['sum', 'count', 'mean'],
        'quantidade': 'sum' if 'quantidade' in df.columns else lambda x: len(x)
    }).round(2)
    
    product_metrics.columns = ['Total_Vendas', 'Qtd_Vendas', 'Ticket_Medio', 'Qtd_Produtos']
    product_metrics = product_metrics.sort_values('Total_Vendas', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 produtos por receita
        top_products = product_metrics.head(10)
        fig = px.bar(x=top_products['Total_Vendas'], y=top_products.index,
                    orientation='h', title='Top 10 Produtos - Receita Total',
                    labels={'x': 'Receita (R$)', 'y': 'Produto'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Matriz de desempenho (Receita vs Quantidade)
        fig = px.scatter(product_metrics, x='Qtd_Produtos', y='Total_Vendas',
                        hover_name=product_metrics.index,
                        title='Matriz de Desempenho: Receita vs Volume',
                        labels={'x': 'Quantidade Vendida', 'y': 'Receita Total (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de categorias se disponível
    if 'categoria' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            category_sales = df.groupby('categoria')['valor'].sum().sort_values(ascending=False)
            fig = px.pie(values=category_sales.values, names=category_sales.index,
                        title='Vendas por Categoria de Produto')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Performance por categoria
            category_metrics = df.groupby('categoria').agg({
                'valor': ['sum', 'mean'],
                'margem_lucro_pct': 'mean' if 'margem_lucro_pct' in df.columns else lambda x: 0
            })
            category_metrics.columns = ['Total_Vendas', 'Ticket_Medio', 'Margem_Media']
            
            fig = px.bar(category_metrics, x=category_metrics.index, y='Margem_Media',
                        title='Margem de Lucro Média por Categoria',
                        labels={'x': 'Categoria', 'y': 'Margem (%)'})
            st.plotly_chart(fig, use_container_width=True)

def analise_canal_vendas(df):
    """Análise de Desempenho por Canal de Vendas"""
    st.markdown('<div class="section-header">🏪 Análise de Desempenho por Canal</div>', unsafe_allow_html=True)
    
    if 'canal' not in df.columns:
        st.warning("Dados de canal de vendas não encontrados.")
        return
    
    # Métricas por canal
    channel_metrics = df.groupby('canal').agg({
        'valor': ['sum', 'count', 'mean'],
        'cliente': 'nunique' if 'cliente' in df.columns else lambda x: len(x),
        'margem_lucro_pct': 'mean' if 'margem_lucro_pct' in df.columns else lambda x: 0
    }).round(2)
    
    channel_metrics.columns = ['Total_Vendas', 'Qtd_Vendas', 'Ticket_Medio', 'Clientes_Unicos', 'Margem_Media']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Participação por canal
        fig = px.pie(values=channel_metrics['Total_Vendas'], 
                    names=channel_metrics.index,
                    title='Participação nas Vendas por Canal')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Eficiência por canal (Ticket médio vs Margem)
        fig = px.scatter(channel_metrics, x='Ticket_Medio', y='Margem_Media',
                        size='Total_Vendas', hover_name=channel_metrics.index,
                        title='Eficiência por Canal: Ticket vs Margem',
                        labels={'x': 'Ticket Médio (R$)', 'y': 'Margem (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela resumo
    st.subheader("📋 Resumo por Canal")
    st.dataframe(channel_metrics.style.format({
        'Total_Vendas': 'R$ {:,.0f}',
        'Ticket_Medio': 'R$ {:,.0f}',
        'Margem_Media': '{:.1f}%'
    }))

def analise_margem_lucro(df):
    """Análise de Margem de Lucro"""
    st.markdown('<div class="section-header">💹 Análise de Margem de Lucro</div>', unsafe_allow_html=True)
    
    if 'margem_lucro_pct' not in df.columns:
        st.warning("Dados de custo não encontrados para calcular margem de lucro.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição de margens
        fig = px.histogram(df, x='margem_lucro_pct', nbins=20,
                          title='Distribuição das Margens de Lucro',
                          labels={'x': 'Margem de Lucro (%)', 'y': 'Frequência'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Margem vs Volume
        fig = px.scatter(df, x='valor', y='margem_lucro_pct',
                        title='Margem de Lucro vs Volume de Vendas',
                        labels={'x': 'Volume de Vendas (R$)', 'y': 'Margem (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise por produto
    if 'produto' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            # Top produtos por margem
            product_margin = df.groupby('produto')['margem_lucro_pct'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=product_margin.values, y=product_margin.index,
                        orientation='h', title='Top 10 Produtos - Maior Margem',
                        labels={'x': 'Margem Média (%)', 'y': 'Produto'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Produtos com menor margem
            low_margin = df.groupby('produto')['margem_lucro_pct'].mean().sort_values().head(10)
            fig = px.bar(x=low_margin.values, y=low_margin.index,
                        orientation='h', title='Top 10 Produtos - Menor Margem',
                        labels={'x': 'Margem Média (%)', 'y': 'Produto'})
            st.plotly_chart(fig, use_container_width=True)

def analise_desempenho_regiao(df):
    """Análise de Desempenho por Região"""
    st.markdown('<div class="section-header">🗺️ Análise de Desempenho por Região</div>', unsafe_allow_html=True)
    
    if 'regiao' not in df.columns:
        st.warning("Dados de região não encontrados.")
        return
    
    # Métricas por região
    region_metrics = df.groupby('regiao').agg({
        'valor': ['sum', 'count', 'mean'],
        'cliente': 'nunique' if 'cliente' in df.columns else lambda x: len(x)
    }).round(2)
    
    region_metrics.columns = ['Total_Vendas', 'Qtd_Vendas', 'Ticket_Medio', 'Clientes_Unicos']
    region_metrics = region_metrics.sort_values('Total_Vendas', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendas por região
        fig = px.bar(region_metrics, x=region_metrics.index, y='Total_Vendas',
                    title='Vendas Totais por Região',
                    labels={'x': 'Região', 'y': 'Vendas (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Comparação ticket médio
        fig = px.bar(region_metrics, x=region_metrics.index, y='Ticket_Medio',
                    title='Ticket Médio por Região',
                    labels={'x': 'Região', 'y': 'Ticket Médio (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Mapa de calor se houver dados temporais
    if 'data' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            # Crescimento por região
            region_time = df.groupby(['regiao', df['data'].dt.to_period('M')])['valor'].sum().unstack(fill_value=0)
            if len(region_time.columns) > 1:
                region_growth = ((region_time.iloc[:, -1] - region_time.iloc[:, 0]) / region_time.iloc[:, 0] * 100).sort_values(ascending=False)
                
                fig = px.bar(x=region_growth.values, y=region_growth.index,
                            orientation='h', title='Crescimento por Região (%)',
                            labels={'x': 'Crescimento (%)', 'y': 'Região'})
                st.plotly_chart(fig, use_container_width=True)

def analise_clientes(df):
    """Análise de Clientes"""
    st.markdown('<div class="section-header">👥 Análise de Clientes</div>', unsafe_allow_html=True)
    
    if 'cliente' not in df.columns:
        st.warning("Dados de cliente não encontrados.")
        return
    
    # Métricas por cliente
    customer_metrics = df.groupby('cliente').agg({
        'valor': ['sum', 'count', 'mean'],
        'data': ['min', 'max'] if 'data' in df.columns else lambda x: None
    }).round(2)
    
    customer_metrics.columns = ['Total_Compras', 'Qtd_Compras', 'Ticket_Medio', 'Primeira_Compra', 'Ultima_Compra']
    
    if 'data' in df.columns:
        customer_metrics['Dias_Cliente'] = (customer_metrics['Ultima_Compra'] - customer_metrics['Primeira_Compra']).dt.days
        customer_metrics['Valor_Vida_Cliente'] = customer_metrics['Total_Compras']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 20 clientes
        top_customers = customer_metrics.nlargest(20, 'Total_Compras')
        fig = px.bar(x=top_customers['Total_Compras'], y=top_customers.index,
                    orientation='h', title='Top 20 Clientes por Valor Total',
                    labels={'x': 'Valor Total (R$)', 'y': 'Cliente'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribuição de valor por cliente
        fig = px.histogram(customer_metrics, x='Total_Compras', nbins=20,
                          title='Distribuição do Valor por Cliente',
                          labels={'x': 'Valor Total por Cliente (R$)', 'y': 'Quantidade de Clientes'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise RFM simplificada
    if 'data' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            # Recência vs Frequência
            fig = px.scatter(customer_metrics, x='Qtd_Compras', y='Dias_Cliente',
                            size='Total_Compras', hover_name=customer_metrics.index,
                            title='Análise RFM: Frequência vs Recência',
                            labels={'x': 'Frequência de Compras', 'y': 'Dias como Cliente'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Segmentação de clientes
            customer_segments = pd.cut(customer_metrics['Total_Compras'], 
                                     bins=3, labels=['Bronze', 'Prata', 'Ouro'])
            segment_counts = customer_segments.value_counts()
            
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title='Segmentação de Clientes por Valor')
            st.plotly_chart(fig, use_container_width=True)

def analise_equipe_vendas(df):
    """Análise de Desempenho da Equipe de Vendas"""
    st.markdown('<div class="section-header">👨‍💼 Análise da Equipe de Vendas</div>', unsafe_allow_html=True)
    
    if 'vendedor' not in df.columns:
        st.warning("Dados de vendedor não encontrados.")
        return
    
    # Métricas por vendedor
    seller_metrics = df.groupby('vendedor').agg({
        'valor': ['sum', 'count', 'mean'],
        'cliente': 'nunique' if 'cliente' in df.columns else lambda x: len(x),
        'margem_lucro_pct': 'mean' if 'margem_lucro_pct' in df.columns else lambda x: 0
    }).round(2)
    
    seller_metrics.columns = ['Total_Vendas', 'Qtd_Vendas', 'Ticket_Medio', 'Clientes_Unicos', 'Margem_Media']
    seller_metrics = seller_metrics.sort_values('Total_Vendas', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ranking de vendedores
        fig = px.bar(seller_metrics.head(15), x=seller_metrics.head(15).index, y='Total_Vendas',
                    title='Top 15 Vendedores - Vendas Totais',
                    labels={'x': 'Vendedor', 'y': 'Vendas (R$)'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Eficiência dos vendedores
        fig = px.scatter(seller_metrics, x='Qtd_Vendas', y='Ticket_Medio',
                        size='Total_Vendas', hover_name=seller_metrics.index,
                        title='Eficiência: Qtd Vendas vs Ticket Médio',
                        labels={'x': 'Quantidade de Vendas', 'y': 'Ticket Médio (R$)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise temporal de vendedores
    if 'data' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            # Performance mensal dos top 5 vendedores
            top_5_sellers = seller_metrics.head(5).index
            monthly_seller_data = df[df['vendedor'].isin(top_5_sellers)].groupby([
                df['data'].dt.to_period('M'), 'vendedor'
            ])['valor'].sum().unstack(fill_value=0)
            
            fig = px.line(monthly_seller_data.T, title='Performance Mensal - Top 5 Vendedores',
                         labels={'index': 'Mês', 'value': 'Vendas (R$)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Consistência dos vendedores (coeficiente de variação)
            monthly_variation = df.groupby(['vendedor', df['data'].dt.to_period('M')])['valor'].sum().groupby('vendedor').std() / df.groupby(['vendedor', df['data'].dt.to_period('M')])['valor'].sum().groupby('vendedor').mean()
            monthly_variation = monthly_variation.sort_values().head(10)
            
            fig = px.bar(x=monthly_variation.values, y=monthly_variation.index,
                        orientation='h', title='Top 10 Vendedores Mais Consistentes',
                        labels={'x': 'Coeficiente de Variação', 'y': 'Vendedor'})
            st.plotly_chart(fig, use_container_width=True)

def analise_conversao(df):
    """Análise de Conversão"""
    st.markdown('<div class="section-header">🎯 Análise de Conversão</div>', unsafe_allow_html=True)
    
    # Análise de conversão por canal
    if 'canal' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Taxa de conversão por canal (simulada)
            channel_conv = df.groupby('canal').agg({
                'valor': 'count',
                'cliente': 'nunique' if 'cliente' in df.columns else lambda x: len(x)
            })
            channel_conv['taxa_conversao'] = (channel_conv['valor'] / channel_conv['cliente'] * 100).round(1)
            
            fig = px.bar(channel_conv, x=channel_conv.index, y='taxa_conversao',
                        title='Taxa de Conversão por Canal (%)',
                        labels={'x': 'Canal', 'y': 'Taxa de Conversão (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Funil de conversão
            funnel_data = {
                'Estágio': ['Visitantes', 'Leads', 'Oportunidades', 'Vendas'],
                'Quantidade': [10000, 2000, 500, len(df)]  # Dados simulados
            }
            funnel_df = pd.DataFrame(funnel_data)
            
            fig = px.funnel(funnel_df, x='Quantidade', y='Estágio',
                           title='Funil de Conversão')
            st.plotly_chart(fig, use_container_width=True)

def analise_precos(df):
    """Análise de Preços"""
    st.markdown('<div class="section-header">💰 Análise de Preços</div>', unsafe_allow_html=True)
    
    if 'preco_unitario' not in df.columns:
        st.warning("Dados de preço unitário não disponíveis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição de preços
        fig = px.histogram(df, x='preco_unitario', nbins=20,
                          title='Distribuição dos Preços Unitários',
                          labels={'x': 'Preço Unitário (R$)', 'y': 'Frequência'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Preço vs Volume
        price_volume = df.groupby('preco_unitario')['quantidade'].sum().reset_index()
        fig = px.scatter(price_volume, x='preco_unitario', y='quantidade',
                        title='Elasticidade: Preço vs Volume',
                        labels={'x': 'Preço Unitário (R$)', 'y': 'Volume Vendido'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise por produto
    if 'produto' in df.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            # Produtos mais caros
            expensive_products = df.groupby('produto')['preco_unitario'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=expensive_products.values, y=expensive_products.index,
                        orientation='h', title='Top 10 Produtos - Maior Preço Médio',
                        labels={'x': 'Preço Médio (R$)', 'y': 'Produto'})
            st.plotly_chart(fig, use_container_width=True)

def analise_rentabilidade(df):
    """Análise de Rentabilidade"""
    st.markdown('<div class="section-header">📊 Análise de Rentabilidade</div>', unsafe_allow_html=True)
    
    if 'margem_lucro' not in df.columns:
        st.warning("Dados de custo não disponíveis para análise de rentabilidade.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rentabilidade total
        total_revenue = df['valor'].sum()
        total_cost = df['custo'].sum() if 'custo' in df.columns else 0
        total_profit = df['margem_lucro'].sum()
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        
        st.metric("💰 Receita Total", f"R$ {total_revenue:,.0f}")
        st.metric("💸 Custo Total", f"R$ {total_cost:,.0f}")
        st.metric("💹 Lucro Total", f"R$ {total_profit:,.0f}")
        st.metric("📈 ROI", f"{roi:.1f}%")
    
    with col2:
        # Rentabilidade ao longo do tempo
        if 'data' in df.columns:
            monthly_profit = df.groupby(df['data'].dt.to_period('M'))['margem_lucro'].sum().reset_index()
            monthly_profit['data'] = monthly_profit['data'].astype(str)
            
            fig = px.line(monthly_profit, x='data', y='margem_lucro',
                         title='Evolução da Rentabilidade Mensal',
                         labels={'x': 'Mês', 'y': 'Lucro (R$)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Rentabilidade por dimensões
    col3, col4 = st.columns(2)
    
    with col3:
        # Rentabilidade por produto
        if 'produto' in df.columns:
            product_profit = df.groupby('produto')['margem_lucro'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=product_profit.values, y=product_profit.index,
                        orientation='h', title='Top 10 Produtos - Maior Lucro',
                        labels={'x': 'Lucro (R$)', 'y': 'Produto'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Rentabilidade por canal
        if 'canal' in df.columns:
            channel_profit = df.groupby('canal')['margem_lucro'].sum().sort_values(ascending=False)
            fig = px.pie(values=channel_profit.values, names=channel_profit.index,
                        title='Distribuição do Lucro por Canal')
            st.plotly_chart(fig, use_container_width=True)

def generate_insights(df):
    """Gera insights automáticos"""
    st.markdown('<div class="section-header">🔍 Insights Automáticos</div>', unsafe_allow_html=True)
    
    insights = []
    
    # Insight de crescimento
    if 'data' in df.columns:
        monthly_sales = df.groupby(df['data'].dt.to_period('M'))['valor'].sum()
        if len(monthly_sales) >= 2:
            growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[0]) / monthly_sales.iloc[0] * 100)
            if growth_rate > 0:
                insights.append(f"📈 **Crescimento Positivo**: As vendas cresceram {growth_rate:.1f}% no período analisado.")
            else:
                insights.append(f"📉 **Atenção**: As vendas declinaram {abs(growth_rate):.1f}% no período analisado.")
    
    # Insight de concentração de clientes
    if 'cliente' in df.columns:
        customer_concentration = df.groupby('cliente')['valor'].sum().sort_values(ascending=False)
        top_5_percent = customer_concentration.head(int(len(customer_concentration) * 0.05)).sum()
        total_sales = customer_concentration.sum()
        concentration_rate = (top_5_percent / total_sales * 100)
        
        if concentration_rate > 50:
            insights.append(f"⚠️ **Alta Concentração**: Os top 5% dos clientes representam {concentration_rate:.1f}% das vendas. Considere diversificar a base de clientes.")
        else:
            insights.append(f"✅ **Boa Diversificação**: Os top 5% dos clientes representam {concentration_rate:.1f}% das vendas.")
    
    # Insight de sazonalidade
    if 'mes' in df.columns:
        monthly_avg = df.groupby('mes')['valor'].mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        variation = ((monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100)
        
        if variation > 30:
            insights.append(f"📅 **Forte Sazonalidade**: Variação de {variation:.1f}% entre meses. Melhor mês: {best_month}, Pior mês: {worst_month}.")
    
    # Insight de margem
    if 'margem_lucro_pct' in df.columns:
        avg_margin = df['margem_lucro_pct'].mean()
        if avg_margin > 30:
            insights.append(f"💹 **Excelente Margem**: Margem média de {avg_margin:.1f}% indica boa rentabilidade.")
        elif avg_margin < 10:
            insights.append(f"⚠️ **Margem Baixa**: Margem média de {avg_margin:.1f}% pode indicar necessidade de otimização de custos.")
    
    # Exibir insights
    for insight in insights:
        st.markdown(f'<div class="highlight">{insight}</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📊 Dashboard Avançado de Análise de Vendas</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 Upload de Dados")
    uploaded_file = st.sidebar.file_uploader(
        "Escolha um arquivo Excel (.xlsx, .xls)",
        type=['xlsx', 'xls'],
        help="Faça upload do seu arquivo de vendas"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df_raw = load_data(uploaded_file)
            st.sidebar.success(f"✅ {len(df_raw)} registros carregados!")
            
            # Detectar colunas
            column_mapping = auto_detect_columns(df_raw)
            
            # Mostrar mapeamento
            if column_mapping:
                st.sidebar.subheader("🔄 Colunas Detectadas")
                for key, value in column_mapping.items():
                    st.sidebar.text(f"{value} → {key}")
            
            # Processar dados
            df = process_data(df_raw, column_mapping)
            
            # Filtros
            st.sidebar.subheader("🔍 Filtros")
            
            # Filtro de período
            if 'data' in df.columns:
                date_range = st.sidebar.date_input(
                    "Período de Análise",
                    value=(df['data'].min().date(), df['data'].max().date()),
                    min_value=df['data'].min().date(),
                    max_value=df['data'].max().date()
                )
                
                if len(date_range) == 2:
                    df = df[(df['data'].dt.date >= date_range[0]) & 
                           (df['data'].dt.date <= date_range[1])]
            
            # Outros filtros dinâmicos
            filter_columns = ['produto', 'vendedor', 'canal', 'regiao', 'categoria']
            for col in filter_columns:
                if col in df.columns:
                    unique_values = df[col].unique()
                    selected_values = st.sidebar.multiselect(
                        f"{col.capitalize()}",
                        options=unique_values,
                        default=unique_values
                    )
                    if selected_values:
                        df = df[df[col].isin(selected_values)]
            
            # Seleção de análises
            st.sidebar.subheader("📊 Análises Disponíveis")
            analyses = {
                "KPIs Principais": True,
                "Tendências Temporais": st.sidebar.checkbox("Tendências Temporais", True),
                "Desempenho por Produto": st.sidebar.checkbox("Desempenho por Produto", True),
                "Canal de Vendas": st.sidebar.checkbox("Canal de Vendas", 'canal' in df.columns),
                "Margem de Lucro": st.sidebar.checkbox("Margem de Lucro", 'margem_lucro_pct' in df.columns),
                "Desempenho por Região": st.sidebar.checkbox("Desempenho por Região", 'regiao' in df.columns),
                "Análise de Clientes": st.sidebar.checkbox("Análise de Clientes", 'cliente' in df.columns),
                "Equipe de Vendas": st.sidebar.checkbox("Equipe de Vendas", 'vendedor' in df.columns),
                "Análise de Conversão": st.sidebar.checkbox("Análise de Conversão"),
                "Análise de Preços": st.sidebar.checkbox("Análise de Preços", 'preco_unitario' in df.columns),
                "Rentabilidade": st.sidebar.checkbox("Rentabilidade", 'margem_lucro' in df.columns),
                "Insights Automáticos": st.sidebar.checkbox("Insights Automáticos", True)
            }
            
            # Dashboard principal
            if not df.empty:
                # KPIs sempre visíveis
                create_kpi_dashboard(df)
                st.divider()
                
                # Análises selecionadas
                if analyses["Tendências Temporais"]:
                    analise_tendencias_temporais(df)
                    st.divider()
                
                if analyses["Desempenho por Produto"]:
                    analise_desempenho_produto(df)
                    st.divider()
                
                if analyses["Canal de Vendas"] and 'canal' in df.columns:
                    analise_canal_vendas(df)
                    st.divider()
                
                if analyses["Margem de Lucro"] and 'margem_lucro_pct' in df.columns:
                    analise_margem_lucro(df)
                    st.divider()
                
                if analyses["Desempenho por Região"] and 'regiao' in df.columns:
                    analise_desempenho_regiao(df)
                    st.divider()
                
                if analyses["Análise de Clientes"] and 'cliente' in df.columns:
                    analise_clientes(df)
                    st.divider()
                
                if analyses["Equipe de Vendas"] and 'vendedor' in df.columns:
                    analise_equipe_vendas(df)
                    st.divider()
                
                if analyses["Análise de Conversão"]:
                    analise_conversao(df)
                    st.divider()
                
                if analyses["Análise de Preços"] and 'preco_unitario' in df.columns:
                    analise_precos(df)
                    st.divider()
                
                if analyses["Rentabilidade"] and 'margem_lucro' in df.columns:
                    analise_rentabilidade(df)
                    st.divider()
                
                if analyses["Insights Automáticos"]:
                    generate_insights(df)
                    st.divider()
                
                # Dados detalhados
                with st.expander("🔍 Dados Processados"):
                    st.dataframe(df)
                
                # Download
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                st.download_button(
                    label="📥 Download Dados Processados",
                    data=excel_buffer.getvalue(),
                    file_name=f"analise_vendas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            else:
                st.warning("⚠️ Nenhum dado encontrado após aplicar os filtros.")
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    
    else:
        # Informações sobre o formato esperado
        st.info("👆 Faça upload de um arquivo Excel para começar as análises avançadas")
        
        st.subheader("📋 Colunas Recomendadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Colunas Obrigatórias:**
            - 📅 Data das vendas
            - 💰 Valor das vendas
            
            **Colunas Opcionais:**
            - 🛍️ Produto/Item
            - 👨‍💼 Vendedor
            - 🏪 Canal de Vendas
            - 🗺️ Região/Localização
            """)
        
        with col2:
            st.markdown("""
            **Para Análises Avançadas:**
            - 👥 Cliente
            - 💸 Custo do produto
            - 📦 Quantidade vendida
            - 🏷️ Categoria do produto
            - 🎯 Tipo de cliente
            """)
        
        # Exemplo de dados
        st.subheader("📊 Exemplo de Formato de Dados")
        example_data = pd.DataFrame({
            'Data': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Produto': ['Notebook Dell', 'Mouse Logitech', 'Teclado Mecânico'],
            'Valor': [2500.00, 89.90, 299.90],
            'Custo': [1800.00, 45.00, 150.00],
            'Quantidade': [1, 2, 1],
            'Vendedor': ['João Silva', 'Maria Santos', 'Pedro Costa'],
            'Canal': ['E-commerce', 'Loja Física', 'E-commerce'],
            'Region': ['SP', 'RJ', 'MG'],
            'Cliente': ['Empresa XYZ', 'João Paulo', 'Tech Corp'],
            'Categoria': ['Informática', 'Acessórios', 'Acessórios']
        })
        
        st.dataframe(example_data)

if __name__ == "__main__":
    main()
