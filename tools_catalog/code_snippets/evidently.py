import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Function to detect data drift
def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, column_mapping: ColumnMapping = None):
    # Create a report with the DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    
    # Run the report with the provided data
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    
    # Return the results
    return report.show()

# Example usage
if __name__ == "__main__":
    # Example reference and current data
    reference_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })

    current_data = pd.DataFrame({
        'feature1': [1, 2, 3, 3, 5],
        'feature2': [5, 3, 3, 2, 2]
    })

    # Detect data drift
    detect_data_drift(reference_data, current_data)