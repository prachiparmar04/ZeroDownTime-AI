import streamlit as st 
import logging
from crewapp import runapp

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom CSS for button styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: navy;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'insights_generated' not in st.session_state:
    st.session_state.insights_generated = False

def generate_insights():
    st.session_state.insights_generated = True

def format_insights(insights_text):
    # Remove the initial heading
    insights_text = insights_text.replace("🔍 **Final Summary Report:**", "")
    
    # Split by periods and clean up each point
    # points = [point.strip() for point in insights_text.split('.') if point.strip()]
    
    # # Format as markdown bullet points
    # formatted_output = ""
    # for point in points:
    #     if point:  # Only add non-empty points
    #         formatted_output += f"{point}.\n\n"
    
    return insights_text
path = "/Users/prachiparmar/Documents/zerodowntime_ai_agentic/src/zerodowntime_ai_agentic/logo5.png"
st.image(path)
def main():
    st.title("Machine Insights Generator")
    st.subheader("Click the button below to generate insights from machine data.")
    
    # Logo

    
    # Create form to better control execution
    with st.form("insights_form"):
        st.form_submit_button("GENERATE MACHINE INSIGHTS", on_click=generate_insights)
    
    # Output area
    output_placeholder = st.empty()
    
    # Only run when form is submitted
    if st.session_state.insights_generated:
        try:
            with st.status("⏳ Generating insights... Please wait", expanded=True) as status:
                insights = runapp()
                formatted_insights = format_insights(insights)
                output_placeholder.markdown(formatted_insights)
               
                status.update(label="✅ Insights Generated!", state="complete")
        except Exception as e:
            logging.error(f"Error generating machine insights: {e}")
            st.error("Error generating machine insights")
        finally:
            # Reset the state
            st.session_state.insights_generated = False

if __name__ == "__main__":
    main()