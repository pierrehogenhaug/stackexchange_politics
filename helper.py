import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define columns for each DataFrame
comment_cols = ['Id', 'PostId', 'Score', 'Text', 'CreationDate', 'UserId']
post_cols = [
    'Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 
    'Score', 'ViewCount', 'Body', 'OwnerUserId', 'LastActivityDate', 
    'Title', 'Tags', 'AnswerCount', 'CommentCount'
]
post_links_cols = ['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId']
tags_cols = ['Id', 'TagName', 'Count']
user_cols = ['Id', 'Reputation', 'CreationDate', 'LastAccessDate', 'Views', 'UpVotes', 'DownVotes']

# Load DataFrames
df_comments = pd.read_csv('./output/Comments.csv', usecols=comment_cols)
df_posts = pd.read_excel('./output/Posts.xlsx', usecols=post_cols)
df_posts_links = pd.read_csv('./output/PostLinks.csv', usecols=post_links_cols)
df_tags = pd.read_csv('./output/Tags.csv', usecols=tags_cols)
df_users = pd.read_csv('./output/Users.csv', usecols=user_cols)

# Typecasting
df_comments = df_comments.astype({
    'Id': 'int32',
    'PostId': 'int32',
    'Score': 'int16',
    'Text': 'object',
    'CreationDate': 'datetime64[ns]',
    'UserId': 'int32'
})

df_posts = df_posts.astype({
    'Id': 'int32',
    'PostTypeId': 'int32',
    'ParentId': 'int32',
    'AcceptedAnswerId': 'int16',
    'CreationDate': 'datetime64[ns]',
    'Score': 'int16',
    'ViewCount': 'int32',
    'Body': 'object',
    'OwnerUserId': 'int32',
    'LastActivityDate': 'datetime64[ns]',
    'Title': 'string',
    'Tags': 'string',
    'AnswerCount': 'int16',
    'CommentCount': 'int16'
})

df_posts_links = df_posts_links.astype({
    'Id': 'int32',
    'CreationDate': 'datetime64[ns]',
    'PostId': 'int32',
    'RelatedPostId': 'int32',
    'LinkTypeId': 'uint8'
})

df_tags = df_tags.astype({
    'Id': 'int32',
    'TagName': 'string',
    'Count': 'int32'
    #'ExcerptPostId': 'int32',
    #'WikiPostId': 'int32'
})

df_users = df_users.astype({
    'Id': 'int32',
    'Reputation': 'int32',
    'CreationDate': 'datetime64[ns]',
    #'DisplayName': 'string',
    'LastAccessDate': 'datetime64[ns]',
    'Views': 'int32',
    'UpVotes': 'int32',
    'DownVotes': 'int32',
})

# Aggregate Post and Comment Counts per User to find the active users (posts+comments >= 40)
