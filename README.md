# On-Route Charging Dispenser Explorer

## Overview

This project provides a simple tool to analyze **on-route charging
demand for Battery Electric Buses (BEBs)** and estimate how many
**pantograph chargers (dispensers)** are needed at each terminal
location.

The dashboard helps planners understand charger demand and evaluate how
charger capacity affects bus operations.

## What the Tool Shows

For each charging location, the tool provides:

-   Planned number of dispensers
-   Total charging sessions
-   Number of unique bus blocks

It also visualizes:

-   Charging session timeline (Gantt chart)
-   Peak charging demand per minute
-   Coverage curves showing how many blocks or sessions can be supported
    as charger numbers increase

These insights help identify **terminal congestion and charging
infrastructure needs**.

## Data Inputs

`charging_event_record_test.csv`\
Charging session records.

`dispensers_needed_by_candidate.csv`\
Planned number of chargers for each location.

## Run the App

Install dependencies:

pip install streamlit pandas plotly

Run the dashboard:

streamlit run dispenser_explorer.py

## License

MIT License
