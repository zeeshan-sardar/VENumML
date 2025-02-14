# %%
"""
## Graph Analytics Demo
"""

# %%
"""
In this demo we will using encrypted graphs and PageRank to show how employee reviews can be mined for key insights, all while keeping sensitive information private. The scenario is as follows: 

* Each employee is invited to review 5 of their colleagues that they have directly worked with. 
* Each review allows an employee to rate their colleagues an integer value in the range [1,5], where 1 is extremely negative, and 5 is extremely postive.
* In this simplified scenario a dataframe is created via employee submissions. The dataframe is then used to calculate average ratings for each employee, which are then encrypted. Additionally, a network graph is created. In this graph, each employee forms a node, and the directed, weighted edge between two employees symbolizes a review.
* Using pagerank, one is able to securely determine key individuals that are vital to the organization: these are people who work many other key employees, who in turn work with lots of other employees. In this sense we are able to enrich the average rating by also ascertaining how integrated an employee is, and therefore wether they are a critical asset to the organization, or even a liability.  

"""

# %%
# Import Dependencies
import sys
sys.path.append('..')


# %%
import pandas as pd
import numpy as np
import random
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
# pio.renderers.default = 'notebook'
# pio.renderers.default = 'iframe'


# %%
from venumML.venumpy import small_glwe as vp
from venumML.graphs.venum_graph import *

from venumML.venum_tools import *

# %%
ctx = vp.SecretContext()
ctx.precision = 6

# %%
"""
Let's import the sensitive dataframe of 360 employee reviews.
"""

# %%
reviews = pd.read_csv('/Users/zeeshan.sardar/VENumML/demos/use_cases/graph_demo/data/reviews.csv')

# %%
reviews.head(10)


# %%
"""
Now, let's convert the dataframe into an encrypted graph. The "from node" will be the Reviewer. The "to node" will be the Reviewee. The rating assigned by the Reviewer will act as the weight. In this case, for simplicity, we won't be hashing the nodes.
"""

# %%
G = df_to_encrypted_graph(ctx, reviews,from_col='Reviewer',to_col='Reviewee', weight_col= None, use_hashing=False)

# %%
"""
Next, let's calculate the average rating per employee, and store it in a nested dictionary, along with their role. 
"""

# %%
# Create a dictionary where the employee is the key and reviewer_role is the value
employee_roles = reviews.set_index('Reviewer')['reviewer_role'].to_dict()

# Calculate the average score per employee, including those who didn't get a review
average_scores = reviews.groupby('Reviewee').apply(lambda x: {'AverageScore': x['Score'].mean(), 'Role': x['reviewer_role'].iloc[0]}).to_dict()

# Add employees who didn't get a review with a score of 0
for employee in employee_roles:
    if employee not in average_scores:
        average_scores[employee] = {'AverageScore': 0, 'Role': employee_roles[employee]}

average_scores

# %%
"""
Before transmitting this data for analysis, let's encrypt the data using the secret context we created earlier.
"""

# %%
for k in average_scores.keys():
    average_scores[k]['AverageScore'] = ctx.encrypt(average_scores[k]['AverageScore'])

# %%
average_scores

# %%
"""
To keep things streamlined, we will continue the notebook as if we were now in an separate environment, and had imported the encrypted graph and dictionary sent to us.
"""

# %%
"""
Let's now run the encrypted graph through our pagerank algorithm. 
"""

# %%
encrypted_pagerank = pagerank(ctx, G, iters=20)

# %%
encrypted_pagerank

# %%
"""
Now, let's decrypt the PageRank scores.
"""

# %%
decrypted_scores = decrypt_pagerank(ctx, encrypted_pagerank)

# %%
decrypted_scores

# %%
"""
Let's also decrypt the average review ratings.
"""

# %%
# Decrypt the average scores
decrypted_average_scores = {k: {'AverageScore': ctx.decrypt(v['AverageScore']), 'Role': v['Role']} for k, v in average_scores.items()}

# Convert the decrypted average scores to a DataFrame
average_scores_df = pd.DataFrame.from_dict(decrypted_average_scores, orient='index')


# %%
"""
Next, we'd like to merge that with the pagerank scores to make it simple to plot and analyze our data
"""

# %%
# Create DataFrames from average_scores and decrypted_pagerank
decrypted_pagerank_df = pd.DataFrame.from_dict(decrypted_scores, orient='index', columns=['PageRank'])

# Merge the two DataFrames on the index (employee name)
performance_metrics = average_scores_df.merge(decrypted_pagerank_df, left_index=True, right_index=True)

# Reset the index to make employee names a column
performance_metrics.reset_index(inplace=True)
performance_metrics.rename(columns={'index': 'Employee'}, inplace=True)

performance_metrics.head()


# %%
"""
The PageRank combined with the Average rating can be used to cluster employees into three groups of high impact, medium impact and low impact employees.
"""

# %%
# Perform K-Means clustering
num_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
performance_metrics['Cluster'] = kmeans.fit_predict(performance_metrics[['PageRank', 'AverageScore']])


# %%
"""
Now let's plot the results! In the below graph, the employees pagerank is represented by the size of their marker. The colour indicates the average rating that they recieved, and the shape symbolizes the cluster they belong to. On the x-axis we have the PageRank, on the y axis we have the average rating.
"""

# %%
# Set marker sizes based on PageRank, scaled for visualization
pagerank_min, pagerank_max = performance_metrics['PageRank'].min(), performance_metrics['PageRank'].max()
performance_metrics['MarkerSize'] = 10 + 50 * ((performance_metrics['PageRank'] - pagerank_min) / (pagerank_max - pagerank_min))


# Define the cmin and cmax for consistent color scaling with the network graph
score_min = performance_metrics['AverageScore'].min()
score_max = performance_metrics['AverageScore'].max()

# Plot the clusters using Plotly, with AverageScore for color, PageRank for size, and Role for symbol
fig = px.scatter(
    performance_metrics,
    x='PageRank',
    y='AverageScore',
    color='AverageScore',
    size='MarkerSize',
    color_continuous_scale='RdBu',  # Same color scale as the network graph, reversed
    symbol='Cluster',  # Different markers for each role
    hover_data={
        'Employee': True,
        'PageRank': ':.4f',
        'AverageScore': ':.2f',
        'Role': True,  # Display the role in hover text
        'MarkerSize': False  # Hide MarkerSize in hover text
    },
    title="Employee Impact Based on PageRank and Average Rating",
    labels={'PageRank': 'PageRank', 'AverageScore': 'Average Rating'}
)

# Indent the title to the right
fig.update_layout(title={'x': 0.5})  

# Apply cmin and cmax for consistent color range with the network graph
fig.update_traces(marker=dict(colorbar=dict(title="Average Rating"), cmin=score_min, cmax=score_max))

# Hide the Role legend
fig.update_layout(
    showlegend=False
)

# Show the plot
fig.show()


# %%
"""
Let's examine the above graph.
* Noah Rhodes is the highest ranking employee according to PageRank. This makes sense. He's an executive and interfaces with many employees at the company. According to his average score he seems to be well-liked.
* Sherry Decker is an intern, and unfortunately is the lowest ranked. This is because nobody reviewed Sherry, highlighting a flaw in the 360 review system. HR should solicite reviews on her behalf, as her current score of 0 is not even a valid option. Meanwhile, Anthony Rodriguez has been recognized for a very successful internship.
* Jamie Arnold is a high ranking executive who is scoring poorly on his 360 performance reviews. Their peformance may require further examination by the CEO.
"""

# %%
"""

"""