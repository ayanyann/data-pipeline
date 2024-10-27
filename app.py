import streamlit as st
import pandas as pd
import unicodedata
import re
from textblob import TextBlob

def enhanced_normalize_text(text):
    
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Define wider range of mathematical and fancy Unicode characters
    # Mathematical bold (𝐀-𝐙, 𝐚-𝐳)
    
    bold_chars = dict(zip(range(0x1D400, 0x1D433), list(range(65, 91)) + list(range(97, 123))))
    # Mathematical italic (𝘈-𝘡, 𝘢-𝘻)
    italic_chars = dict(zip(range(0x1D608, 0x1D63B), list(range(65, 91)) + list(range(97, 123))))
    # mathematical variants
    math_chars = dict(zip(range(0x1D434, 0x1D467), list(range(65, 91)) + list(range(97, 123))))
    
    # combine all character mappings
    all_chars = {**bold_chars, **italic_chars, **math_chars}
    translation = str.maketrans({chr(k): chr(v) for k, v in all_chars.items()})
    
    # convert known mathematical unicode to regular letters
    text = text.translate(translation)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def label_disaster_sentiment(text):

    if pd.isna(text):
        return 0
        
    text = str(text).lower()
    
    # negative keywords with pinoy terms
    negative_keywords = [
        
        # Weather warnings and alerts
        'warning', 'alert', 'advisory', 'bulletin', 'severe', 'intensity',
        'heavy rainfall', 'monsoon', 'thunderstorm', 'lightning', 'flood',
        'bagyo', 'bagyong', 'ulan', 'delubyo', 'lindol', 'kidlat', 'baha',
        
        # Damage and danger related
        'damage', 'damaged', 'destroy', 'destroyed', 'casualties', 'victim',
        'injured', 'injury', 'dead', 'death', 'fatality', 'trapped', 'stranded',
        'missing', 'lost', 'nasira', 'nawala', 'nasalanta', 'namatay',
        
        # Emergency situations
        'emergency', 'danger', 'dangerous', 'critical', 'disaster', 'calamity',
        'crisis', 'accident', 'incident', 'crash', 'collision', 'hazard',
        'threat', 'risk', 'panganib', 'sakuna', 'aksidente',
        
        # Weather conditions
        'storm', 'typhoon', 'cyclone', 'landslide', 'earthquake', 'tsunami',
        'tornado', 'hurricane', 'flash flood', 'overflow', 'surge', 'gale',
        'bagyo', 'habagat', 'amihan', 'daluyong', 'pagbaha',
        
        # Actions taken
        'evacuation', 'evacuate', 'suspend', 'suspended', 'cancel', 'cancelled',
        'close', 'closed', 'lockdown', 'shutdown', 'stop', 'halt', 'block',
        'ban', 'restrict', 'isolate', 'quarantine', 'isolasyon', 'sarado',
        
        # Impact descriptions
        'affected', 'displaced', 'devastated', 'devastating', 'severe', 'worst',
        'massive', 'extensive', 'catastrophic', 'dire', 'grave', 'serious',
        'malubha', 'malala', 'matindi', 'apektado'
    ]
    
    #  positive keywords 
    positive_keywords = [
        # Response and rescue
        'rescue', 'rescued', 'save', 'saved', 'recover', 'recovered', 'survival',
        'survivor', 'alive', 'found', 'safe', 'secure', 'protect', 'protected',
        'salvage', 'retrieve', 'evacuated', 'nagligtas', 'nasagip', 'nailigtas',
        
        # Assistance and support
        'assist', 'assistance', 'help', 'support', 'aid', 'relief', 'donate',
        'donation', 'contribute', 'contribution', 'provide', 'provision',
        'supply', 'distribute', 'distribution', 'tulong', 'ayuda', 'donasyon',
        
        # Recovery and improvement
        'improve', 'improvement', 'restore', 'restoration', 'rebuild', 'repair',
        'fix', 'fixed', 'maintain', 'maintenance', 'upgrade', 'strengthen',
        'reinforce', 'rehabilitate', 'pagbangon', 'pagkumpuni', 'ayos',
        
        # Preparation and readiness
        'prepare', 'preparation', 'ready', 'readiness', 'alert', 'vigilant',
        'monitor', 'monitoring', 'watch', 'observe', 'observation', 'check',
        'inspect', 'inspection', 'paghahanda', 'handang-handa', 'alerto',
        
        # Response teams and operations
        'team', 'unit', 'force', 'group', 'crew', 'staff', 'personnel',
        'volunteer', 'worker', 'responder', 'officer', 'official', 'authority',
        'coordinator', 'coordination', 'kagawad', 'tauhan', 'boluntaryo',
        
        # Positive outcomes
        'success', 'successful', 'accomplish', 'achievement', 'complete',
        'completion', 'progress', 'advance', 'improvement', 'better', 'good',
        'positive', 'effective', 'tagumpay', 'matagumpay', 'maganda',
        
        # Community response
        'community', 'together', 'unity', 'united', 'cooperate', 'cooperation',
        'coordinate', 'coordination', 'collaborate', 'collaboration', 'partner',
        'partnership', 'bayanihan', 'sama-sama', 'tulong-tulong'
    ]
    
    # Count keyword matches
    negative_count = sum(1 for keyword in negative_keywords if keyword in text)
    positive_count = sum(1 for keyword in positive_keywords if keyword in text)
    
    # Use TextBlob for basic sentiment analysis
    blob_sentiment = TextBlob(text).sentiment.polarity
    
    # Combined rule-based and ML approach with adjusted weights
    if negative_count > positive_count + 1:  # requiring stronger negative signal
        return -1
    elif positive_count > negative_count:
        return 1
    elif blob_sentiment < -0.1:
        return -1
    elif blob_sentiment > 0.1:
        return 1
    else:
        return 0

st.title("Disaster Data Processor")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame preview
    st.write("Dataset Preview:")
    st.write(df.head())
    
    # Sidebar for column selection and processing options
    st.sidebar.header("Processing Options")
    primary_column = st.sidebar.selectbox("Select primary column:", df.columns)
    
    # Text processing options
    text_processing = st.sidebar.expander("Text Processing Options")
    normalize_text = text_processing.checkbox("Normalize text (remove special characters and formatting)", value=False)
    analyze_sentiment = text_processing.checkbox("Analyze sentiment", value=False)
    
    # Column combination options
    column_options = st.sidebar.expander("Column Options")
    combine_columns = column_options.checkbox("Combine with another column to fill empty rows", value=False)
    remove_empty_rows = column_options.checkbox("Remove empty rows in the new column", value=False)
    
    # Process the data based on user selections
    if combine_columns:
        fallback_column = column_options.selectbox("Select column to combine it:", df.columns)
        new_column_name = column_options.text_input("Enter new column name for combined data:", 
                                                  value="Combined_Column")
        
        if primary_column != fallback_column:
            # Create combined column
            df[new_column_name] = df[primary_column].fillna(df[fallback_column])
            working_column = new_column_name
        else:
            st.sidebar.error("Primary and fallback columns cannot be the same.")
            working_column = primary_column
    else:
        working_column = primary_column
    
    # Apply text normalization if selected
    if normalize_text:
        normalized_column_name = f"normalized_{working_column}"
        df[normalized_column_name] = df[working_column].apply(enhanced_normalize_text)
        working_column = normalized_column_name
    
    # Apply sentiment analysis if selected
    if analyze_sentiment:
        # Add sentiment columns
        df['sentiment'] = df[working_column].apply(label_disaster_sentiment)
        df['sentiment_category'] = df['sentiment'].map({
            -1: 'negative',
            0: 'neutral',
            1: 'positive'
        })
    
        # Display sentiment distribution
        st.write("### Sentiment Distribution")
        sentiment_counts = df['sentiment_category'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Remove empty rows if selected
        if remove_empty_rows:
            df = df[df[working_column].notna()]
        
        # Display processed data
        st.write("### Processed Data Preview")
        display_columns = [working_column]
        if analyze_sentiment:
            display_columns.extend(['sentiment', 'sentiment_category'])
        st.write(df[display_columns].head(10))
        # Add download button for processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    

else:
    st.write("Please upload a CSV file to continue.")
