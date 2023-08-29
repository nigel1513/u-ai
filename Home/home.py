import streamlit as st
from streamlit_lottie import st_lottie
import json

def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_coding = load_lottiefile("Home/animation_ll5tbz57.json")
statistics_coding = load_lottiefile("Home/statistics.json")
analysis_coding = load_lottiefile("Home/analysis.json")
viz_coding = load_lottiefile("Home/visualization.json")
ai_coding = load_lottiefile("Home/ai.json")



#lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ldulgcir.json")



# Define the HTML for the navigation bar
nav_html = """
<nav class="top-nav">
    <div class="left">
        <a href="#" style="font-family: 'Helvetica', sans-serif; font-weight: bold; font-size: 20px;">U&AI</a>
    </div>
    <div class="right">
        <a href="#">Work</a>
        <a href="#">About</a>
    </div>
</nav>
"""

# Define the CSS for the navigation bar
nav_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300&display=swap');
.top-nav {
    background-color: white;
    color: black;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    font-family: 'Raleway', sans-serif;
    height: 50px;
}

.top-nav .left {
    margin-left: -250px;
}

.top-nav .left a {
    text-decoration: none;
    color: black;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 2px;
    margin-left: 15px;
    transition: color 0.3s;
}

.top-nav .right {
    margin-right: -350px;
}

.top-nav .right a {
    text-decoration: none;
    color: black;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 2px;
    margin-left: 15px;
    transition: color 0.3s;
}

.top-nav .right a:hover {
    color: gray;
}

</style>
"""

main_content_html = """
<div class="main-content">
    <div class="welcome-text">
        <h1>Welcome to U&AI</h1>
        <p>We are a team of AI enthusiasts...</p>
    </div>
    <div class="decorative-line"></div>
</div>
"""

# Define the CSS for the main content
main_content_css = """
<style>
.main-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 50px;
    background-color: white; /* 배경색을 검은색으로 변경 */
    font-family: 'Raleway', sans-serif;
    width: 100%;
    margin: 0 auto;
}

.welcome-text {
    text-align: center;
    margin-bottom: 30px;
}

.welcome-text h1 {
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 10px;
}

.welcome-text p {
    font-size: 18px;
    line-height: 1.5;
}

.decorative-line {
    width: 50px;
    height: 3px;
    background-color: #333;
}
</style>
"""


def home_app():
    # Display the navigation bar and main content
    st.markdown(nav_css, unsafe_allow_html=True)
    st.markdown(nav_html, unsafe_allow_html=True)

    # Display the main content
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.markdown(main_content_css, unsafe_allow_html=True)
    st.markdown(main_content_html, unsafe_allow_html=True)
    
    st.write('\n')    
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    
    # Display the Work section
    # Display the Work section
    st.markdown("""
    <div class="work-section" style="text-align: center;">
        <h2 class="work-title">Work</h2>
    </div>
    """, unsafe_allow_html=True)
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.markdown("""
    <div class="work-section" style="text-align: center;">
        <h2 class="work-title">Statistics</h2>
    </div>
    """, unsafe_allow_html=True)

    # Display the Statistics section

    st_lottie(
    statistics_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
    height=400,
    width=None,
    key=None,
    )   
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    
    st.markdown("""
    <div class="work-section" style="text-align: center;">
        <h2 class="work-title">Pre-Processing</h2>
    </div>
    """, unsafe_allow_html=True)

    st_lottie(
    analysis_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
    height=500,
    width=None,
    key=None,
    )   
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    
    st.markdown("""
    <div class="work-section" style="text-align: center;">
        <h2 class="work-title">Visualization</h2>
    </div>
    """, unsafe_allow_html=True)


    st_lottie(
    viz_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
    height=400,
    width=None,
    key=None,
    )   

    # Display the Machine Learning section
    st.write('\n')   
    st.write('\n')
    st.write('\n')   
    st.write('\n')
    st.write('\n')

    st.markdown("""
    <div class="work-section" style="text-align: center;">
        <h2 class="work-title">Machine Learning</h2>
    </div>
    """, unsafe_allow_html=True)



    st_lottie(
    ai_coding,
    speed=3,
    reverse=False,
    loop=True,
    quality="high",
    height=600,
    width=None,
    key=None,
    )   


    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    


    footer_html = """
    <div class="footer">
        <div class="footer-content">
            <p>U&AI is by Design Yu.</p>
            <p>Copyright © 2023 Design Yu. All rights reserved.</p>
            <p>Made with from Korea</p>
        </div>
        <div class="footer-links">
            <a href="https://github.com/nigel1513"><i class="fab fa-github" style="font-size: 25px;"></i></a>
            <a href="#"><i class="fab fa-youtube" style="font-size: 25px;"></i></a>
            <a href="#"><i class="fab fa-discord" style="font-size: 25px;"></i></a>
            <a href="#"><i class="fas fa-user" style="font-size: 25px;"></i></a>
        </div>
    </div>
    """

    footer_html = """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    """ + footer_html

    footer_css = """
    <style>
    .footer {
        background-color: #f8f8f8;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        margin-left: -250px;
        align-items: center;
        width: 185%; /* Adjust the width as needed */
    }

    .footer p {
        margin: 0;
    }

    .footer-links {
        display: flex;
        gap: 20px;
    }

    .footer-links a {
        text-decoration: none;
        color: #333;
        font-weight: bold;
    }

    </style>
    """

    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)