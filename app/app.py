# app.py

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_data(filepath):
    file_extension = filepath.rsplit('.', 1)[1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    return df


def get_frequent_itemsets(df, transaction_column, item_column, min_support=0.01):
    # Limit the number of transactions to process
    max_transactions = 5000  # Reduced from 10000
    if len(df) > max_transactions:
        df = df.sample(max_transactions, random_state=42)
    
    transactions = df.groupby(transaction_column)[item_column].apply(list).tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets['items_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    
    # Limit the number of itemsets
    frequent_itemsets = frequent_itemsets.head(50)  # Reduced from 100
    
    print("frequent_itemsets columns:", frequent_itemsets.columns)
    
    return frequent_itemsets


def get_association_rules(frequent_itemsets, min_threshold=0.5):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    rules = rules.sort_values('confidence', ascending=False)
    
    # Limit the number of rules
    rules = rules.head(25)  # Reduced from 50
    
    return rules


def format_itemset(itemset):
    return ', '.join(list(itemset))


def create_support_visualization(frequent_itemsets):
    # Take only top 5 itemsets for visualization
    top_itemsets = frequent_itemsets.head(5)  # Reduced from 10
    
    fig = px.bar(top_itemsets,
                 x='items_str',
                 y='support',
                 title='Top 5 Frequent Itemsets',
                 labels={'items_str': 'Itemset', 'support': 'Support'},
                 color='support',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat='.2%',
        height=300,  # Reduced from 400
        margin=dict(t=30, b=50)  # Reduced margins
    )
    return fig.to_html(full_html=False)


def create_confidence_visualization(rules):
    # Create a copy with string representations
    rules_copy = rules.copy()
    rules_copy['antecedents_str'] = rules_copy['antecedents'].apply(format_itemset)
    rules_copy['consequents_str'] = rules_copy['consequents'].apply(format_itemset)
    
    # Take only top 10 rules for visualization
    rules_copy = rules_copy.head(10)  # Reduced from 20
    
    fig = px.scatter(rules_copy,
                     x='confidence',
                     y='lift',
                     size='support',
                     color='lift',
                     title='Top 10 Association Rules',
                     labels={'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'},
                     hover_data=['antecedents_str', 'consequents_str', 'support'],
                     color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_tickformat='.2%',
        height=300,  # Reduced from 400
        margin=dict(t=30, b=30)  # Reduced margins
    )
    return fig.to_html(full_html=False)


def create_itemset_network(rules):
    # Take only top 8 rules for network visualization
    rules = rules.head(8)  # Reduced from 15
    
    nodes = set()
    edges = []
    
    for _, rule in rules.iterrows():
        antecedent = format_itemset(rule['antecedents'])
        consequent = format_itemset(rule['consequents'])
        nodes.add(antecedent)
        nodes.add(consequent)
        edges.append((antecedent, consequent, rule['confidence'], rule['lift']))
    
    # Create node positions
    node_positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / len(nodes)
        node_positions[node] = (np.cos(angle), np.sin(angle))
    
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=edge[2] * 2, color='rgba(0,0,255,0.3)'),  # Reduced width
            hoverinfo='text',
            text=f"Confidence: {edge[2]:.2%}<br>Lift: {edge[3]:.2f}",
            showlegend=False
        ))
    
    # Add nodes
    for node in nodes:
        x, y = node_positions[node]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            text=[node],
            textposition="bottom center",
            marker=dict(size=12, color='blue'),  # Reduced size
            hoverinfo='text',
            textfont=dict(size=7),  # Reduced font size
            showlegend=False
        ))
    
    fig.update_layout(
        title='Top 8 Item Relationships',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10,l=5,r=5,t=30),  # Reduced margins
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300  # Reduced from 400
    )
    
    return fig.to_html(full_html=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['filepath'] = filepath
        try:
            df = process_data(filepath)
            session['columns'] = df.columns.tolist()
            return redirect(url_for('configure'))
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))


@app.route('/configure')
def configure():
    columns = session.get('columns')
    if not columns:
        flash('Please upload a file first')
        return redirect(url_for('index'))
    return render_template('configure.html', columns=columns)


@app.route('/analyze', methods=['POST'])
def analyze():
    filepath = session.get('filepath')
    if not filepath:
        flash('Please upload a file first')
        return redirect(url_for('index'))

    transaction_column = request.form.get('transaction_column')
    item_column = request.form.get('item_column')
    min_support = float(request.form.get('min_support', 1)) / 100
    min_confidence = float(request.form.get('min_confidence', 50)) / 100

    try:
        # Process data with progress indicator
        flash('Processing data...')
        
        df = process_data(filepath)
        frequent_itemsets = get_frequent_itemsets(df, transaction_column, item_column, min_support)
        rules = get_association_rules(frequent_itemsets, min_confidence)

        # Calculate summary statistics
        summary_stats = {
            'total_transactions': len(df[transaction_column].unique()),
            'total_items': len(df[item_column].unique()),
            'total_rules': len(rules),
            'avg_support': frequent_itemsets['support'].mean(),
            'avg_confidence': rules['confidence'].mean(),
            'avg_lift': rules['lift'].mean()
        }

        # Create visualizations
        flash('Creating visualizations...')
        support_plot = create_support_visualization(frequent_itemsets)
        confidence_plot = create_confidence_visualization(rules)
        network_plot = create_itemset_network(rules)

        # Prepare session-safe versions with limited data
        session['frequent_itemsets'] = []
        for _, row in frequent_itemsets.head(20).iterrows():
            items_value = row['items_str']
            # If items_value is not a string, try to convert it
            if not isinstance(items_value, str):
                try:
                    items_value = ', '.join(list(items_value))
                except Exception as e:
                    print('DEBUG conversion error:', e)
                    items_value = str(items_value)
            print('DEBUG FINAL items_value:', items_value, type(items_value))
            session['frequent_itemsets'].append({
                'items': items_value,
                'support': f"{row['support']:.2%}",
                'length': row['length']
            })

        session['association_rules'] = [
            {
                'antecedent': format_itemset(row['antecedents']),
                'consequent': format_itemset(row['consequents']),
                'support': f"{row['support']:.2%}",
                'confidence': f"{row['confidence']:.2%}",
                'lift': f"{row['lift']:.2f}"
            }
            for _, row in rules.head(20).iterrows()  # Limit to 20 rules
        ]

        return render_template('results.html',
                           frequent_itemsets=session['frequent_itemsets'],
                           association_rules=session['association_rules'],
                           support_plot=support_plot,
                           confidence_plot=confidence_plot,
                           network_plot=network_plot,
                           summary_stats=summary_stats)

    except Exception as e:
        flash(f'Error analyzing data: {str(e)}')
        return redirect(url_for('configure'))


@app.route('/item_recommendations', methods=['GET', 'POST'])
def item_recommendations():
    association_rules = session.get('association_rules')
    if not association_rules:
        flash('Please analyze your data first')
        return redirect(url_for('configure'))

    if request.method == 'POST':
        selected_item = request.form.get('selected_item')
        recommendations = [
            {'item': rule['consequent'], 'confidence': rule['confidence'], 'lift': rule['lift']}
            for rule in association_rules if selected_item and selected_item in rule['antecedent']
        ]
        return render_template('recommendations.html', selected_item=selected_item, recommendations=recommendations)

    items = set()
    for rule in association_rules:
        for item in rule['antecedent'].split(', '):
            items.add(item)

    return render_template('item_selector.html', items=sorted(list(items)))


@app.route('/results')
def results():
    if 'frequent_itemsets' not in session or 'association_rules' not in session:
        flash('Please analyze your data first')
        return redirect(url_for('configure'))

    frequent_itemsets = session['frequent_itemsets']
    association_rules = session['association_rules']

    # Provide default summary_stats to avoid Jinja2 error
    summary_stats = {
        'total_transactions': '-',
        'total_items': '-',
        'total_rules': '-',
        'avg_support': 0,
        'avg_confidence': 0,
        'avg_lift': 0
    }
    if 'summary_stats' in session:
        summary_stats = session['summary_stats']

    # Provide empty plots if not available
    support_plot = session.get('support_plot', '')
    confidence_plot = session.get('confidence_plot', '')
    network_plot = session.get('network_plot', '')

    return render_template(
        'results.html',
        frequent_itemsets=frequent_itemsets,
        association_rules=association_rules,
        support_plot=support_plot,
        confidence_plot=confidence_plot,
        network_plot=network_plot,
        summary_stats=summary_stats
    )

# File: app.py

@app.route('/restock_recommendations')
def restock_recommendations():
    recommendations = []
    for itemset in session.get('frequent_itemsets', []):
        if itemset['length'] > 1:
            readable_items = itemset['items']  # Use the string directly
            support_value = itemset['support']

            if isinstance(support_value, str):
                support_value = support_value.replace('%', '')  # remove '%' if present
                support_value = float(support_value) / 100      # divide back to decimal

            recommendations.append({
                'items': readable_items,
                'support': f"{support_value * 100:.2f}%"  # format it again
            })
    return render_template('restock.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
