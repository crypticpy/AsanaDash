import streamlit as st
import asana
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration
class Config(BaseModel):
    ASANA_API_TOKEN: str
    PORTFOLIO_GID: str
    TEAM_GID: str

config = Config(
    ASANA_API_TOKEN=os.getenv("ASANA_API_TOKEN"),
    PORTFOLIO_GID="1207425243280062",
    TEAM_GID="1199588008197352"
)

# Asana API setup
configuration = asana.Configuration()
configuration.access_token = config.ASANA_API_TOKEN
api_client = asana.ApiClient(configuration)

# Initialize Asana API instances
portfolios_api = asana.PortfoliosApi(api_client)
projects_api = asana.ProjectsApi(api_client)
tasks_api = asana.TasksApi(api_client)
sections_api = asana.SectionsApi(api_client)

st.set_page_config(layout="wide")
st.title("Asana Portfolio Dashboard")

# Helper function to handle API errors
def api_error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except asana.rest.ApiException as e:
            st.error(f"API Error in {func.__name__}: {e}")
            return None
    return wrapper

@st.cache_data(ttl=3600)
@api_error_handler
def get_portfolio_projects() -> List[Dict[str, Any]]:
    opts = {
        'opt_fields': 'name,gid',
    }
    return list(portfolios_api.get_items_for_portfolio(config.PORTFOLIO_GID, opts=opts))

@st.cache_data(ttl=3600)
@api_error_handler
def get_tasks(project_gid: str) -> List[Dict[str, Any]]:
    opts = {
        'opt_fields': 'name,completed,due_on,created_at,completed_at,assignee.name,memberships.section.name,custom_fields,tags,num_subtasks',
    }
    return list(tasks_api.get_tasks_for_project(project_gid, opts=opts))

@st.cache_data(ttl=3600)
@api_error_handler
def get_sections(project_gid: str) -> List[Dict[str, Any]]:
    opts = {
        'opt_fields': 'name',
    }
    return list(sections_api.get_sections_for_project(project_gid, opts=opts))

def safe_get(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

def estimate_project_completion(df):
    project_estimates = []

    for project_name, project_df in df.groupby('project'):
        # Calculate average completion time for completed tasks
        completed_tasks = project_df[project_df['status'] == 'Completed']
        avg_completion_time = (completed_tasks['completed_at'] - completed_tasks['created_at']).mean()

        # Count remaining tasks
        remaining_tasks = project_df[project_df['status'] != 'Completed']
        num_remaining_tasks = len(remaining_tasks)

        # Estimate completion time
        estimated_completion_time = avg_completion_time * num_remaining_tasks

        # Get project due date if available
        project_due_date = project_df['project_due_date'].iloc[0] if 'project_due_date' in project_df.columns else None

        # Calculate estimated completion date
        current_date = datetime.now()
        estimated_completion_date = current_date + estimated_completion_time

        # Compare with due date
        days_difference = None
        if project_due_date:
            days_difference = (estimated_completion_date - project_due_date).days

        project_estimates.append({
            'project': project_name,
            'avg_task_completion_time': avg_completion_time,
            'remaining_tasks': num_remaining_tasks,
            'estimated_completion_time': estimated_completion_time,
            'estimated_completion_date': estimated_completion_date,
            'project_due_date': project_due_date,
            'days_difference': days_difference
        })

    return pd.DataFrame(project_estimates)

def process_tasks(tasks: List[Dict[str, Any]], project_name: str) -> List[Dict[str, Any]]:
    processed_tasks = []
    for task in tasks:
        task_data = {
            'project': project_name,
            'name': safe_get(task, 'name'),
            'status': 'Completed' if safe_get(task, 'completed') else 'In Progress',
            'due_date': safe_get(task, 'due_on'),
            'created_at': safe_get(task, 'created_at'),
            'completed_at': safe_get(task, 'completed_at'),
            'assignee': safe_get(task, 'assignee', 'name') or 'Unassigned',
            'section': safe_get(task, 'memberships', 0, 'section', 'name') or 'No section',
            'tags': [tag['name'] for tag in safe_get(task, 'tags') or []],
            'num_subtasks': safe_get(task, 'num_subtasks') or 0,
        }

        # Process custom fields
        custom_fields = safe_get(task, 'custom_fields') or []
        for field in custom_fields:
            field_name = safe_get(field, 'name')
            if field_name:
                task_data[f"custom_{field_name}"] = safe_get(field, 'display_value')

        processed_tasks.append(task_data)
    return processed_tasks


def create_interactive_timeline(project_completion_estimates):
    today = pd.Timestamp.now().floor('D')
    future_placeholder_date = today + pd.DateOffset(years=2)  # Set a placeholder date 2 years in the future

    # Sort projects by estimated completion date
    project_completion_estimates = project_completion_estimates.sort_values('estimated_completion_date')

    fig = go.Figure()

    # Assign a unique y value to each project to avoid overlap
    y_positions = list(range(len(project_completion_estimates)))
    project_completion_estimates['y'] = y_positions

    # Add markers for each project
    for i, (_, project) in enumerate(project_completion_estimates.iterrows()):
        completion_date = project['estimated_completion_date']
        y_position = project['y']

        if pd.isnull(completion_date):
            color = 'red'
            symbol = 'x'
            hover_text = f"{project['project']}<br>Completion date undetermined"
            marker_date = future_placeholder_date
        else:
            if completion_date < today:
                color = 'lightgreen'
                symbol = 'circle'
                hover_text = f"{project['project']}<br>Completed: {completion_date.strftime('%Y-%m-%d')}"
            elif completion_date == today:
                color = 'yellow'
                symbol = 'star'
                hover_text = f"{project['project']}<br>Due today: {completion_date.strftime('%Y-%m-%d')}"
            else:
                color = 'orange'
                symbol = 'diamond'
                hover_text = f"{project['project']}<br>Estimated completion: {completion_date.strftime('%Y-%m-%d')}"
            marker_date = completion_date

        fig.add_trace(go.Scatter(
            x=[marker_date],
            y=[y_position],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol=symbol),
            text=project['project'],
            textposition='top center',
            name=project['project'],
            hoverinfo='text',
            hovertext=hover_text
        ))

    # Add vertical line for today
    fig.add_shape(type="line", x0=today, y0=-1, x1=today, y1=len(y_positions),
                  line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=today, y=len(y_positions), text="Today", showarrow=False, yshift=10)

    # Configure layout
    fig.update_layout(
        title="Project Timeline",
        xaxis=dict(
            title="Date",
            tickformat='%Y-%m-%d',
            range=[today - pd.Timedelta(days=60), future_placeholder_date],
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title="Projects",
            tickvals=y_positions,
            ticktext=project_completion_estimates['project'],
            range=[-1, len(y_positions)]
        ),
        height=600,
        showlegend=False,
        hovermode='closest'
    )

    return fig


# Automatically select all projects from the portfolio
portfolio_projects = get_portfolio_projects()
if portfolio_projects:
    selected_projects = [project['name'] for project in portfolio_projects]
else:
    st.error("Failed to fetch portfolio projects. Please check your API connection.")
    st.stop()

if selected_projects:
    all_tasks = []
    for project_name in selected_projects:
        project = next((p for p in portfolio_projects if p['name'] == project_name), None)
        if project:
            project_gid = project['gid']
            tasks = get_tasks(project_gid)
            if tasks:
                all_tasks.extend(process_tasks(tasks, project_name))
            else:
                st.warning(f"No tasks found for project: {project_name}")
        else:
            st.warning(f"Project not found: {project_name}")

    if all_tasks:
        df = pd.DataFrame(all_tasks)
        df['due_date'] = pd.to_datetime(df['due_date'], utc=True)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['completed_at'] = pd.to_datetime(df['completed_at'], utc=True)

        # Overall Portfolio Metrics
        st.header("Overall Portfolio Metrics")
        total_tasks = len(df)
        completed_tasks = df['status'].value_counts().get('Completed', 0)
        completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed Tasks", completed_tasks)
        col3.metric("Completion Rate", f"{completion_rate:.2f}%")

        # Task Status Distribution
        project_completion_estimates = estimate_project_completion(df)

        st.subheader("Task Status Distribution")
        try:
            fig_status = px.pie(df, names='status', title="Task Status Distribution")
            st.plotly_chart(fig_status)
        except Exception as e:
            st.error(f"Error rendering Task Status Distribution: {e}")

        # Project Estimate Timeline
        st.subheader("Project Timeline")

        # Create and display the timeline
        timeline_fig = create_interactive_timeline(project_completion_estimates)
        st.plotly_chart(timeline_fig, use_container_width=True)


        # Project Completion Estimate Cards
        # Custom CSS for styling
        st.markdown("""
        <style>
            .project-card {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            .project-title {
                font-size: 1.2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .project-status {
                font-size: 1rem;
                font-weight: bold;
                padding: 0.2rem 0.5rem;
                border-radius: 3px;
                display: inline-block;
            }
            .status-on-track {
                background-color: #c8e6c9;
                color: #2e7d32;
            }
            .status-off-track {
                background-color: #ffcdd2;
                color: #c62828;
            }
            .project-detail {
                margin-bottom: 0.2rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Project Completion Estimation
        st.header("Project Completion Estimates")

        try:
            # Calculate estimates


            # Define fiscal year end date
            fiscal_year_end = pd.Timestamp('2024-09-30')

            # Add on-track status column
            project_completion_estimates['on_track'] = project_completion_estimates['estimated_completion_date'].apply(
                lambda x: x <= fiscal_year_end if pd.notnull(x) else False
            )

            # Sort the DataFrame: on-track projects first, then by estimated completion date
            project_completion_estimates = project_completion_estimates.sort_values(
                ['on_track', 'estimated_completion_date'],
                ascending=[False, True]
            )


            # Function to create a card for each project
            def create_project_card(project):
                on_track = project['on_track']
                status_class = "status-on-track" if on_track else "status-off-track"
                status_text = "On Track" if on_track else "Off Track"

                card_html = f"""
                <div class='project-card'>
                    <div class='project-title'>{project['project']}</div>
                    <div class='project-status {status_class}'>{status_text}</div>
                    <div class='project-detail'>Estimated completion: {project['estimated_completion_date'].strftime('%Y-%m-%d') if pd.notnull(project['estimated_completion_date']) else 'Not available'}</div>
                    <div class='project-detail'>Remaining tasks: {project['remaining_tasks']}</div>
                    <div class='project-detail'>Avg task time: {f"{project['avg_task_completion_time'].days:.1f} days" if pd.notnull(project['avg_task_completion_time']) else 'Not available'}</div>
                """
                if project['days_difference'] is not None:
                    status = "ahead of" if project['days_difference'] < 0 else "behind"
                    card_html += f"<div class='project-detail'>{abs(project['days_difference'])} days {status} schedule</div>"
                card_html += "</div>"
                return card_html


            # Create a grid layout for project cards
            for i in range(0, len(project_completion_estimates), 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(create_project_card(project_completion_estimates.iloc[i]), unsafe_allow_html=True)
                if i + 1 < len(project_completion_estimates):
                    with col2:
                        st.markdown(create_project_card(project_completion_estimates.iloc[i + 1]),
                                    unsafe_allow_html=True)

            # Visualize estimates
            st.subheader("Estimated Project Completion Times")
            valid_estimates = project_completion_estimates.dropna(subset=['estimated_completion_time'])

            if not valid_estimates.empty:
                # Create a categorical scale for estimated completion time
                def categorize_time(days):
                    if pd.isnull(days) or days > 365:
                        return "Over a year"
                    elif days <= 30:
                        return "Within 30 days"
                    elif days <= 90:
                        return "1-3 months"
                    elif days <= 180:
                        return "3-6 months"
                    else:
                        return "6-12 months"


                valid_estimates['time_category'] = valid_estimates['estimated_completion_time'].dt.days.apply(
                    categorize_time)

                # Order the categories (reversed order)
                category_order = ["Over a year", "6-12 months", "3-6 months", "1-3 months", "Within 30 days"]

                # Create a custom color scale
                color_scale = {'Within 30 days': 'green',
                               '1-3 months': 'yellowgreen',
                               '3-6 months': 'yellow',
                               '6-12 months': 'orange',
                               'Over a year': 'red'}

                # Create the plot
                fig_completion = px.bar(valid_estimates,
                                        x='project',
                                        y='time_category',
                                        color='time_category',
                                        category_orders={"time_category": category_order},
                                        title="Estimated Time to Project Completion",
                                        labels={'project': 'Project', 'time_category': 'Estimated Timeframe'},
                                        color_discrete_map=color_scale)

                # Update layout
                fig_completion.update_layout(
                    height=500,
                    xaxis_title="Project",
                    yaxis_title="Estimated Timeframe",
                    xaxis_tickangle=-45,
                    legend_title="Estimated Timeframe",
                    yaxis={'categoryorder': 'array', 'categoryarray': category_order}  # Ensure correct order on y-axis
                )

                # Add hover information
                fig_completion.update_traces(
                    hovertemplate="<b>%{x}</b><br>Timeframe: %{y}<extra></extra>"
                )

                st.plotly_chart(fig_completion, use_container_width=True)

                # Add a table with detailed information
                st.subheader("Detailed Project Estimates")
                detailed_estimates = valid_estimates[['project', 'estimated_completion_time', 'time_category']]
                detailed_estimates['estimated_completion_time'] = detailed_estimates[
                    'estimated_completion_time'].dt.days
                detailed_estimates = detailed_estimates.sort_values('estimated_completion_time')
                st.dataframe(detailed_estimates)

            else:
                st.warning("No valid completion time estimates available for visualization.")

            # Summary metrics
            st.subheader("Overall Portfolio Status")
            total_projects = len(project_completion_estimates)
            on_track_projects = sum(project_completion_estimates['on_track'])
            total_remaining_tasks = project_completion_estimates['remaining_tasks'].sum()
            avg_completion_time = project_completion_estimates['avg_task_completion_time'].mean()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Projects", total_projects)
            col2.metric("On Track Projects", f"{on_track_projects} / {total_projects}")
            col3.metric("Total Remaining Tasks", total_remaining_tasks)
            if pd.notnull(avg_completion_time):
                col4.metric("Avg Task Completion", f"{avg_completion_time.days:.1f} days")
            else:
                col4.metric("Avg Task Completion", "N/A")

        except Exception as e:
            st.error(f"Error calculating project completion estimates: {e}")
            st.write("Error details:", str(e))
            import traceback

            st.write("Traceback:", traceback.format_exc())


        # Task Status Over Time
        st.subheader("Task Status Over Time")
        try:
            # Assuming the fiscal year starts on October 1st
            fiscal_year_start = pd.Timestamp('2023-10-01', tz='UTC')
            fiscal_year_end = pd.Timestamp('2024-09-30', tz='UTC')
            current_date = pd.Timestamp.now(tz='UTC')

            # Filter data for the current fiscal year
            df_fiscal = df[(df['created_at'] >= fiscal_year_start) & (df['created_at'] <= fiscal_year_end)]

            # Create a date range for each month in the fiscal year, up to the current month
            date_range = pd.date_range(start=fiscal_year_start, end=min(fiscal_year_end, current_date), freq='MS', tz='UTC')

            avg_completion_time = []
            tasks_created = []
            tasks_overdue = []
            task_details = []

            for start_date in date_range:
                end_date = min(start_date + pd.offsets.MonthEnd(1), current_date)

                # Tasks created this month
                created_mask = (df_fiscal['created_at'] >= start_date) & (df_fiscal['created_at'] <= end_date)
                created_count = created_mask.sum()
                tasks_created.append(created_count)

                # Average completion time for tasks completed this month
                completed_mask = (df_fiscal['completed_at'] >= start_date) & (df_fiscal['completed_at'] <= end_date)
                completed_tasks = df_fiscal[completed_mask]
                avg_time = (completed_tasks['completed_at'] - completed_tasks['created_at']).mean().total_seconds() / 86400  # Convert to days
                avg_completion_time.append(avg_time if not pd.isna(avg_time) else 0)

                # Overdue tasks as of the end of this month
                overdue_mask = (
                    (df_fiscal['due_date'] <= end_date) &
                    (df_fiscal['due_date'].notna()) &
                    (df_fiscal['status'] != 'Completed')
                )
                overdue_count = overdue_mask.sum()
                tasks_overdue.append(overdue_count)

                # Detailed overdue tasks information
                overdue_tasks = df_fiscal[overdue_mask].copy()
                overdue_tasks['days_overdue'] = (end_date - overdue_tasks['due_date']).dt.days
                overdue_tasks_details = overdue_tasks[['name', 'project', 'days_overdue']].sort_values('days_overdue', ascending=False)

                # Store task details for this month
                task_details.append({
                    'month': start_date.strftime('%B %Y'),
                    'avg_completion_time': f"{avg_time:.2f}" if not pd.isna(avg_time) else 'N/A',
                    'tasks_created': created_count,
                    'tasks_overdue': overdue_count,
                    'overdue_tasks': overdue_tasks_details.to_dict('records')
                })

            # Create the plot
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(
                x=date_range,
                y=avg_completion_time,
                mode='lines+markers',
                name='Avg Completion Time (days)',
                customdata=task_details
            ))
            fig.add_trace(go.Scatter(
                x=date_range,
                y=tasks_created,
                mode='lines+markers',
                name='Tasks Created',
                customdata=task_details
            ))
            fig.add_trace(go.Scatter(
                x=date_range,
                y=tasks_overdue,
                mode='lines+markers',
                name='Overdue Tasks',
                customdata=task_details
            ))

            # Update layout
            fig.update_layout(
                title='Task Status Over Time',
                xaxis_title='Month',
                yaxis_title='Value',
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Show the plot
            st.plotly_chart(fig, use_container_width=True)

            # Display some overall statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Completion Time (days)", f"{sum(avg_completion_time) / len(avg_completion_time):.2f}")
            col2.metric("Total Tasks Created", sum(tasks_created))
            col3.metric("Current Overdue Tasks", tasks_overdue[-1])

            # Add a selectbox for month selection
            selected_month = st.selectbox("Select a month for details:", [d['month'] for d in task_details])

            # Update the display of details for the selected month
            if selected_month:
                month_data = next(d for d in task_details if d['month'] == selected_month)

                st.subheader(f"Details for {month_data['month']}")
                st.write(f"Average Completion Time: {month_data['avg_completion_time']} days")
                st.write(f"Tasks Created: {month_data['tasks_created']}")
                st.write(f"Overdue Tasks: {month_data['tasks_overdue']}")

                if month_data['tasks_overdue'] > 0:
                    st.subheader("Overdue Tasks")
                    overdue_df = pd.DataFrame(month_data['overdue_tasks'])
                    st.dataframe(overdue_df)

                    # Group overdue tasks by project
                    project_overdue = overdue_df.groupby('project').agg({
                        'name': 'count',
                        'days_overdue': 'mean'
                    }).reset_index()
                    project_overdue.columns = ['Project', 'Number of Overdue Tasks', 'Average Days Overdue']
                    project_overdue = project_overdue.sort_values('Number of Overdue Tasks', ascending=False)

                    st.subheader("Overdue Tasks by Project")
                    st.dataframe(project_overdue)

                    # Visualize overdue tasks by project
                    fig_overdue = px.bar(project_overdue,
                                         x='Project',
                                         y='Number of Overdue Tasks',
                                         color='Average Days Overdue',
                                         title="Overdue Tasks by Project",
                                         labels={'Average Days Overdue': 'Avg Days Overdue'},
                                         color_continuous_scale='Reds')
                    st.plotly_chart(fig_overdue)

        except Exception as e:
            st.error(f"Error rendering Task Status Over Time: {e}")
            st.error(f"Exception details: {str(e)}")

        # Tasks by Section
        st.subheader("Tasks by Section")
        try:
            # Initialize a dictionary to store all unique sections and their task counts
            section_task_counts = {}
            total_tasks = 0

            for project in portfolio_projects:
                project_gid = project['gid']

                # Fetch sections for the project
                opts = {
                    'opt_fields': 'name,gid'
                }
                sections = list(sections_api.get_sections_for_project(project_gid, opts=opts))

                # Create a mapping of section GIDs to names
                section_map = {section['gid']: section['name'] for section in sections}

                # Fetch tasks for the project
                tasks = get_tasks(project_gid)

                # Count tasks in each section
                for task in tasks:
                    total_tasks += 1
                    memberships = safe_get(task, 'memberships')
                    if memberships:
                        section_gid = safe_get(memberships[0], 'section', 'gid')
                        if section_gid and section_gid in section_map:
                            section_name = section_map[section_gid]
                            section_task_counts[section_name] = section_task_counts.get(section_name, 0) + 1
                        else:
                            section_task_counts['No Section'] = section_task_counts.get('No Section', 0) + 1
                    else:
                        section_task_counts['No Section'] = section_task_counts.get('No Section', 0) + 1

            if section_task_counts:
                # Convert the dictionary to a DataFrame for easier plotting
                df_sections = pd.DataFrame(list(section_task_counts.items()), columns=['Section', 'Task Count'])
                df_sections = df_sections.sort_values('Task Count', ascending=False)

                # Create the bar chart
                fig_sections = px.bar(df_sections,
                                      x='Section',
                                      y='Task Count',
                                      title="Tasks by Section Across All Projects",
                                      labels={'Section': 'Section Name', 'Task Count': 'Number of Tasks'},
                                      color='Task Count',
                                      color_continuous_scale='Viridis')

                fig_sections.update_layout(
                    xaxis_title="Section Name",
                    yaxis_title="Number of Tasks",
                    xaxis_tickangle=-45,
                    height=600,
                    margin=dict(b=100)
                )

                st.plotly_chart(fig_sections, use_container_width=True)

                # Summary statistics
                total_sections = len(df_sections)
                avg_tasks_per_section = total_tasks / total_sections if total_sections > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tasks", total_tasks)
                col2.metric("Total Sections", total_sections)
                col3.metric("Avg Tasks per Section", f"{avg_tasks_per_section:.2f}")

            else:
                st.warning("No sections or tasks found in the projects.")

        except Exception as e:
            st.error(f"Error rendering Tasks by Section: {e}")

        # Tasks by Due Date
        st.subheader("Tasks by Due Date (Oct 1, 2023 - Current Date)")
        try:
            # Define the date range
            start_date = pd.Timestamp('2023-10-01', tz='UTC')
            end_date = pd.Timestamp.now(tz='UTC').floor('D')  # Current date, floored to start of day

            # Filter out tasks without due dates and outside the specified range
            df_with_dates = df[
                (df['due_date'].notna()) &
                (df['due_date'] >= start_date) &
                (df['due_date'] <= end_date)
                ].copy()

            if not df_with_dates.empty:
                # Convert due_date to date only (without time)
                df_with_dates['due_date'] = df_with_dates['due_date'].dt.date

                # Create a pivot table for the stacked bar chart
                pivot_data = df_with_dates.pivot_table(
                    values='name',
                    index='due_date',
                    columns='status',
                    aggfunc='count',
                    fill_value=0
                ).reset_index()

                # Ensure both 'Completed' and 'In Progress' columns exist
                if 'Completed' not in pivot_data.columns:
                    pivot_data['Completed'] = 0
                if 'In Progress' not in pivot_data.columns:
                    pivot_data['In Progress'] = 0

                # Calculate cumulative sums
                pivot_data['Total_Cumulative'] = pivot_data['Completed'].cumsum() + pivot_data['In Progress'].cumsum()
                pivot_data['Completed_Cumulative'] = pivot_data['Completed'].cumsum()
                pivot_data['Remaining_Tasks'] = pivot_data['Total_Cumulative'] - pivot_data['Completed_Cumulative']

                # Create the stacked bar chart
                fig_due_dates = go.Figure()

                fig_due_dates.add_trace(go.Bar(
                    x=pivot_data['due_date'],
                    y=pivot_data['Completed'],
                    name='Completed',
                    marker_color='green'
                ))

                fig_due_dates.add_trace(go.Bar(
                    x=pivot_data['due_date'],
                    y=pivot_data['In Progress'],
                    name='In Progress',
                    marker_color='orange'
                ))

                # Add the line chart for remaining tasks
                fig_due_dates.add_trace(go.Scatter(
                    x=pivot_data['due_date'],
                    y=pivot_data['Remaining_Tasks'],
                    mode='lines',
                    name='Remaining Tasks',
                    line=dict(color='red', width=2)
                ))

                fig_due_dates.update_layout(
                    title="Tasks by Due Date and Remaining Tasks",
                    xaxis_title="Due Date",
                    yaxis_title="Number of Tasks",
                    barmode='stack',
                    legend_title="Task Status",
                    height=500,  # Adjust height as needed
                    xaxis_range=[start_date, end_date]  # Set x-axis range to current date
                )

                st.plotly_chart(fig_due_dates, use_container_width=True)

                # Additional statistics
                total_tasks = len(df_with_dates)
                completed_tasks = df_with_dates['status'].value_counts().get('Completed', 0)
                in_progress_tasks = df_with_dates['status'].value_counts().get('In Progress', 0)
                remaining_tasks = pivot_data['Remaining_Tasks'].iloc[-1]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Tasks in Date Range", total_tasks)
                col2.metric("Completed Tasks", completed_tasks)
                col3.metric("In Progress Tasks", in_progress_tasks)
                col4.metric("Remaining Tasks", int(remaining_tasks))

                # Show tasks due soon (next 30 days)
                st.subheader("Tasks Due Soon (Next 30 Days)")
                today = pd.Timestamp.now(tz='UTC').floor('D')
                thirty_days_from_now = today + pd.Timedelta(days=30)
                tasks_due_soon = df[
                    (df['due_date'].notna()) &  # Ensure due_date is not null
                    (df['due_date'] >= today) &
                    (df['due_date'] <= thirty_days_from_now)
                    ].sort_values('due_date')

                if not tasks_due_soon.empty:
                    st.dataframe(tasks_due_soon[['name', 'due_date', 'status', 'assignee', 'project']])

                    # Add summary statistics
                    total_due_soon = len(tasks_due_soon)
                    completed_due_soon = tasks_due_soon['status'].value_counts().get('Completed', 0)
                    in_progress_due_soon = tasks_due_soon['status'].value_counts().get('In Progress', 0)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Tasks Due Soon", total_due_soon)
                    col2.metric("Completed Tasks", completed_due_soon)
                    col3.metric("In Progress Tasks", in_progress_due_soon)

                    # Add a chart for tasks due soon
                    fig_due_soon = px.bar(
                        tasks_due_soon.groupby('due_date')['name'].count().reset_index(),
                        x='due_date',
                        y='name',
                        title="Tasks Due in the Next 30 Days",
                        labels={'due_date': 'Date', 'name': 'Number of Tasks'}
                    )
                    st.plotly_chart(fig_due_soon)
                else:
                    st.info("No tasks due in the next 30 days.")

            else:
                st.warning("No tasks found with due dates in the specified range (Oct 1, 2023 - Current Date).")

        except Exception as e:
            st.error(f"Error rendering Tasks by Due Date: {e}")
            st.error(f"Exception details: {str(e)}")

        # Assignee Workload
        st.subheader("Assignee Workload")
        try:
            assignee_workload = df[df['status'] != 'Completed']['assignee'].value_counts().reset_index()
            assignee_workload.columns = ['assignee', 'task_count']
            fig_workload = px.bar(assignee_workload, x='assignee', y='task_count',
                                  title="Assignee Workload (Incomplete Tasks)")
            st.plotly_chart(fig_workload)
        except Exception as e:
            st.error(f"Error rendering Assignee Workload: {e}")

        # Time to Completion
        st.subheader("Average Time to Completion")
        try:
            df['completion_time'] = (df['completed_at'] - df['created_at']).dt.total_seconds() / 86400  # Convert to days
            avg_completion_time = df[df['status'] == 'Completed']['completion_time'].mean()
            st.metric("Average Days to Complete a Task", f"{avg_completion_time:.2f}")
        except Exception as e:
            st.error(f"Error calculating Average Time to Completion: {e}")

        # Overdue Tasks
        st.subheader("Overdue Tasks")
        try:
            now = pd.Timestamp.now(tz='UTC')
            overdue_tasks = df[(df['due_date'] < now) & (df['status'] != 'Completed')]
            st.metric("Number of Overdue Tasks", len(overdue_tasks))
            if not overdue_tasks.empty:
                st.dataframe(overdue_tasks[['name', 'due_date', 'assignee']])
        except Exception as e:
            st.error(f"Error rendering Overdue Tasks: {e}")

        # Project Progress
        st.subheader("Project Progress")
        try:
            project_progress = df.groupby('project').agg({
                'status': lambda x: (x == 'Completed').mean()
            }).reset_index()
            project_progress.columns = ['project', 'completion_rate']
            project_progress['completion_percentage'] = project_progress['completion_rate'] * 100

            # Separate completed and ongoing projects
            completed_projects = project_progress[project_progress['completion_percentage'] == 100].sort_values('project')
            ongoing_projects = project_progress[project_progress['completion_percentage'] < 100].sort_values('completion_percentage', ascending=False)

            # Color mapping function
            def get_color(percentage):
                if percentage == 100:
                    return 'green'
                elif percentage > 50:
                    return 'blue'
                elif percentage > 25:
                    return 'orange'
                else:
                    return 'red'

            # Create figure
            fig_progress = go.Figure()

            # Add ongoing projects
            fig_progress.add_trace(go.Bar(
                x=ongoing_projects['project'],
                y=ongoing_projects['completion_percentage'],
                marker_color=[get_color(p) for p in ongoing_projects['completion_percentage']],
                text=ongoing_projects['completion_percentage'].round(1).astype(str) + '%',
                textposition='outside',
                name='Ongoing Projects'
            ))

            # Add completed projects
            if not completed_projects.empty:
                fig_progress.add_trace(go.Bar(
                    x=completed_projects['project'],
                    y=completed_projects['completion_percentage'],
                    marker_color='green',
                    text='100%',
                    textposition='outside',
                    name='Completed Projects'
                ))

            # Update layout
            fig_progress.update_layout(
                title="Project Completion Rate",
                xaxis_title="Project",
                yaxis_title="Completion Percentage",
                yaxis_range=[0, 110],  # Increased to 110 to accommodate labels
                barmode='group',
                separators=',.',
                xaxis={'categoryorder': 'total descending'},  # This will sort the x-axis by bar height
                height=600,  # Increase height to accommodate labels
                margin=dict(t=100)  # Increase top margin to prevent label cutoff
            )

            st.plotly_chart(fig_progress, use_container_width=True)

            # Display projects count horizontally
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completed Projects", len(completed_projects))
            with col2:
                st.metric("Ongoing Projects", len(ongoing_projects))
            with col3:
                st.metric("Total Projects", len(project_progress))

        except Exception as e:
            st.error(f"Error rendering Project Progress: {e}")

        st.subheader("Task Creation vs Completion Over Time")
        try:
            # Define the fiscal year start and end dates
            fiscal_year_start = pd.Timestamp('2023-10-01', tz='UTC')
            fiscal_year_end = pd.Timestamp('2024-09-30', tz='UTC')

            # Ensure 'created_at' and 'completed_at' are in datetime format
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
            df['completed_at'] = pd.to_datetime(df['completed_at'], utc=True)

            # Print data types for debugging
            # st.write("Data types:")
            # st.write(df.dtypes)

            # Print sample data for debugging
            # st.write("Sample data:")
            # st.write(df[['created_at', 'completed_at']].head())

            # Filter the data to include only tasks within the fiscal year
            df_fiscal = df[
                (df['created_at'] >= fiscal_year_start) &
                (df['created_at'] <= fiscal_year_end)
                ].copy()

            # Group by 'created_at' to count task creation
            task_creation = df_fiscal.groupby(df_fiscal['created_at'].dt.date).size().reset_index(name='created')
            task_creation.columns = ['date', 'created']

            # Group by 'completed_at' to count task completion, excluding NaT values
            task_completion = df_fiscal[df_fiscal['completed_at'].notna()].groupby(
                df_fiscal['completed_at'].dt.date).size().reset_index(name='completed')
            task_completion.columns = ['date', 'completed']

            # Merge the creation and completion data on the 'date' column
            task_trends = pd.merge(task_creation, task_completion, on='date', how='outer').fillna(0)

            # Ensure the date range is within the fiscal year
            task_trends = task_trends[
                (task_trends['date'] >= fiscal_year_start.date()) &
                (task_trends['date'] <= fiscal_year_end.date())
                ]

            # Sort the dataframe by date
            task_trends = task_trends.sort_values('date')

            # Plot the trends
            fig_trends = px.line(task_trends, x='date', y=['created', 'completed'],
                                 title="Task Creation vs Completion Over Time",
                                 labels={'value': 'Number of Tasks', 'variable': 'Task Type'})

            # Set the x-axis range to the fiscal year
            fig_trends.update_xaxes(range=[fiscal_year_start, fiscal_year_end])

            st.plotly_chart(fig_trends)

        except Exception as e:
            st.error(f"Error rendering Task Creation vs Completion: {e}")
            st.write("Error details:", str(e))
            st.write("Error type:", type(e).__name__)
            import traceback

            st.write("Traceback:", traceback.format_exc())

        st.subheader("Task Complexity by Project")
        try:
            # Define complexity bins and labels
            complexity_bins = [-1, 0, 1, 3, 5, 10, float('inf')]
            complexity_labels = ['No subtasks', '1 subtask', '2-3 subtasks', '4-5 subtasks', '6-10 subtasks',
                                 '10+ subtasks']

            # Create a custom color scale from green to red
            colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']

            # Categorize tasks by complexity
            df['complexity'] = pd.cut(df['num_subtasks'], bins=complexity_bins, labels=complexity_labels,
                                      include_lowest=True)

            # Group by project and complexity
            complexity_by_project = df.groupby(['project', 'complexity']).size().unstack(fill_value=0)

            # Sort projects by total number of tasks
            complexity_by_project['total'] = complexity_by_project.sum(axis=1)
            complexity_by_project = complexity_by_project.sort_values('total', ascending=False)
            complexity_by_project = complexity_by_project.drop('total', axis=1)

            # Create stacked bar chart
            fig = go.Figure()

            for i, complexity in enumerate(complexity_labels):
                fig.add_trace(go.Bar(
                    y=complexity_by_project.index,
                    x=complexity_by_project[complexity],
                    name=complexity,
                    orientation='h',
                    marker=dict(color=colors[i]),
                    hoverinfo='text',
                    hovertext=[f"{project}<br>{complexity}: {count}" for project, count in
                               zip(complexity_by_project.index, complexity_by_project[complexity])]
                ))

            fig.update_layout(
                title="Task Complexity Distribution by Project",
                barmode='stack',
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Number of Tasks",
                legend_title="Complexity",
                height=600,
                margin=dict(l=200)  # Increase left margin to accommodate project names
            )

            st.plotly_chart(fig)

            # Display summary statistics
            st.subheader("Summary Statistics")
            total_tasks = df['num_subtasks'].count()
            avg_subtasks = df['num_subtasks'].mean()
            max_subtasks = df['num_subtasks'].max()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tasks", total_tasks)
            col2.metric("Average Subtasks per Task", f"{avg_subtasks:.2f}")
            col3.metric("Max Subtasks in a Task", max_subtasks)

        except Exception as e:
            st.error(f"Error rendering Task Complexity by Project: {e}")
            st.write("Error details:", str(e))
            st.write("Error type:", type(e).__name__)
            import traceback

            st.write("Traceback:", traceback.format_exc())

else:
    st.info("No projects found in the portfolio.")

# Run the app with: streamlit run asana_dashboard.py
