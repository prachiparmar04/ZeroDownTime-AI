from crewai import Task, Agent, Crew, LLM
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import os
import io
import re
  
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")


# Watsonx LLM
WATSONX_MODEL_ID = "watsonx/mistralai/mistral-large"
llm = LLM(
    model=WATSONX_MODEL_ID,
    max_tokens=5000,
    temperature=0.7,
    api_key=WATSONX_APIKEY,

)

base_path = '/Users/prachiparmar/Documents/zerodowntime_ai_agentic/src/zerodowntime_ai_agentic/'
DATA_FILE_PATH = base_path+"reading_turbine_1.csv"
OUTPUT_FILE_PATH = base_path+"forecast_output.csv"
# Load pre-trained model
with open(base_path+"model_rf.pickle", "rb") as f:
    model = pickle.load(f)

# LOAD ALL DATA 
equipment_file_path = "/Users/prachiparmar/Documents/zerodowntime_ai_agentic/src/zerodowntime_ai_agentic/equipment_100_records.csv"
maintenance_file_path = "/Users/prachiparmar/Documents/zerodowntime_ai_agentic/src/zerodowntime_ai_agentic/updated_maintenance_schedule.csv"
import pandas as pd
# Read the datasets
equipment_df = pd.read_csv(equipment_file_path)
maintenance_df = pd.read_csv(maintenance_file_path)

# Load forecasted data
input_df = pd.read_csv(DATA_FILE_PATH)
forecasted_df = pd.read_csv(OUTPUT_FILE_PATH)

# ==============================
# AGENT 1: EQUIPMENT ANALYSIS
# ==============================
equipment_analysis_agent = Agent(
    role="Industrial Equipment Analyst",
    goal="Provide a summary of insights on equipment monitoring and maintenance status.",
    backstory="You are an expert in monitoring industrial machinery and ensuring smooth operations. "
              "You analyze operational data and maintenance schedules to detect potential issues before they escalate.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


equipment_analysis_task = Task(
    description=f"""
    Analyze the provided **Equipment Status** and **Maintenance Schedule** to generate key insights.

    **Equipment Data:**
    {equipment_df.to_string(index=False)}

    **Maintenance Data:**
    {maintenance_df.to_string(index=False)}

    **Analysis Requirements:**
    1. Calculate the **percentage of idle equipment**.
    2. Identify **equipment currently under maintenance**.
    3. Detect **anomalies in temperature, pressure, and vibration levels**.
    4. Summarize **efficiency levels** across all equipment.
    5. Identify the **next upcoming maintenance tasks**.
    6. Highlight **any overdue maintenance** based on the current date.

    Provide a **structured summary** of findings.
    """,
    agent=equipment_analysis_agent,
    expected_output="A structured summary report covering equipment monitoring insights and maintenance schedule analysis with timestamps and reasons."
)

# ==============================
# AGENT 2: FORECASTING
# ==============================

# Forecasting Task
def forecasting_task_function():
    sample_data = input_df.head(10)  # Restrict input data to the first 10 rows
    prompt = f"Given the following historical sensor readings, forecast the next 24 hours for temperature, pressure, vibration, and humidity.\n\n{sample_data.to_string(index=False)}"
    response = llm.generate(prompt)
    
    # Extract CSV data from response
    match = re.search(r'```csv\n(.*?)\n```', response, re.DOTALL)
    if match:
        csv_data = match.group(1)
        forecasted_df = pd.read_csv(io.StringIO(csv_data))
        forecasted_df = forecasted_df.head(50)  # Restrict output to 50 rows
        
        # Save forecast to CSV
        forecasted_df.to_csv(OUTPUT_FILE_PATH, index=False)
        return forecasted_df
    else:
        print("No valid CSV data found in response.")
        return None
forecasting_agent = Agent(
    role="Time-Series Forecaster",
    goal="Analyze historical sensor data and forecast the next 24 hours for key parameters.",
    backstory="An AI model specialized in predictive maintenance, leveraging Watsonx AI for forecasting sensor anomalies.",
    allow_delegation=False,
    llm=llm
)

forecasting_task = Task(
    description="Analyze the provided DataFrame and generate a 24-hour forecast for ['temperature', 'pressure', 'vibration', 'humidity']. Output should be a CSV file with columns (['timestamp','temperature', 'pressure', 'vibration', 'humidity']) with a maximum of 50 rows.",
    agent=forecasting_agent,
    expected_output="A CSV comma seperated text containing the predicted values for the next 72 hours, limited to 50 rows.",
    function=forecasting_task_function
)

# ==============================
# AGENT 3: ANOMALY DETECTION
# ==============================
anomaly_detection_agent = Agent(
    role="AI Anomaly Detector",
    backstory="Advanced AI for real-time anomaly detection in IoT-enabled manufacturing.",
    goal="Detect anomalies in forecasted IoT sensor data and generate human-readable alerts.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

def preprocess_data(df):
    """Preprocess IoT sensor data for anomaly detection."""
    df = df.copy()

    # Ensure timestamp is in datetime format and extract time features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def predict_anomalies(forecasted_df):
    features = preprocess_data(forecasted_df)
    features = features.drop(columns=["Unnamed: 0","timestamp"], errors="ignore")
    forecasted_df["predicted_faulty"] = model.predict(features)
    anomalies = forecasted_df[forecasted_df["predicted_faulty"] == 1][:5]
    return anomalies

predicted_df =  predict_anomalies(forecasted_df)

anomaly_detection_task = Task(
    description="Summarize detected anomalies for turbine-1, listing timestamps and sensor values. {predicted_df}",
    agent=anomaly_detection_agent,
    expected_output="Generate text-based anomaly alerts for turbine-1. Each alert should be a one-liner containing the machine name, timestamp, and sensor values (temperature, pressure, vibration, and humidity). Analyze the possible reasons for each anomaly based on deviations from normal operating conditions. Limit the output to a maximum of five anomalies."
)

# ==============================
# AGENT 4: SUMMARIZER
# ==============================
summarizer_agent = Agent(
    role="Operational Insights Summarizer",
    backstory="AI expert in summarizing anomalies, maintenance, and machine insights for decision-making.",
    goal="Generate a structured summary with three distinct sections: machine status, upcoming maintenance, and detected anomalies.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

summarization_task = Task(
    description="Summarize all findings into three clearly defined sections:\n"
                f"1. **Machine Status:** Provide statistics on machines that are running, idle, or under maintenance from {equipment_analysis_task.expected_output}.\n"
                f"2. **Upcoming Maintenance:** Summarize scheduled and urgent maintenance tasks from {equipment_analysis_task.expected_output}.\n"
                f"3. **Anomalies Detected:** List detected anomalies, their severity, and possible causes from {anomaly_detection_task.expected_output}.",
    agent=summarizer_agent,
    expected_output='''
**Machine Status:**  
- Running: [Summary of how many machines are currently operational and their efficiency]  
- Idle: [Summary of idle machines and reasons for idleness]  
- Maintenance: [Machines currently under maintenance and expected downtime]  

**Upcoming Maintenance:**  
- Scheduled: [List of maintenance activities planned with dates]  
- Urgent: [Critical maintenance actions required immediately]  

**Anomalies Detected:**  
- Issues: [Summary of anomalies detected with timestamps and severity]  
- Potential Causes: [Possible explanations for these anomalies]  
'''
)


# ==============================
# CREW EXECUTION ORDER
# ==============================

def runapp():
    crew = Crew(
        agents=[equipment_analysis_agent, forecasting_agent, anomaly_detection_agent, summarizer_agent],
        tasks=[equipment_analysis_task, forecasting_task, anomaly_detection_task, summarization_task],
        process="sequential"  # Ensures Agent 1 runs first, Agents 2 & 3 in parallel, and Agent 4 last.
    )

    summary_report = crew.kickoff()
    print("🔍 **Final Summary Report:**\n", summary_report)
    return summary_report.raw


summary = runapp()
print(summary)