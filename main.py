import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import colorsys
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from styles import get_css, metric_card_html
import streamlit as st
import extra_streamlit_components as stx
import asana

st.set_page_config(layout="wide")
# Function to get cookie manager
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# Function to save config to cookies and session state
def load_config():
    return {
        "ASANA_API_TOKEN": st.session_state.get("ASANA_API_TOKEN", ""),
        "PORTFOLIO_GID": st.session_state.get("PORTFOLIO_GID", ""),
        "TEAM_GID": st.session_state.get("TEAM_GID", "")
    }

def save_config(config):
    for key, value in config.items():
        st.session_state[key] = value




# Load config at the start
config = load_config()

# Configuration
class Config(BaseModel):
    ASANA_API_TOKEN: str
    PORTFOLIO_GID: str
    TEAM_GID: str

# Set up page configuration

st.title("Asana Portfolio Dashboard")

# Sidebar for user input
st.sidebar.title("Asana Configuration")
asana_token = st.sidebar.text_input("Asana API Token", value=st.session_state.get("ASANA_API_TOKEN", ""), type="password")
portfolio_gid = st.sidebar.text_input("Portfolio GID", value=st.session_state.get("PORTFOLIO_GID", ""))
team_gid = st.sidebar.text_input("Team GID", value=st.session_state.get("TEAM_GID", ""))

if st.sidebar.button("Save Configuration"):
    new_config = {
        "ASANA_API_TOKEN": asana_token,
        "PORTFOLIO_GID": portfolio_gid,
        "TEAM_GID": team_gid
    }
    save_config(new_config)
    st.sidebar.success("Configuration saved!")
    st.rerun()


# Check if all required configurations are set
if not all([config.get("ASANA_API_TOKEN"), config.get("PORTFOLIO_GID"), config.get("TEAM_GID")]):
    st.warning("Please enter your Asana API Token, Portfolio GID, and Team GID in the sidebar to continue.")
    st.stop()

# Create Config object
config_obj = Config(
    ASANA_API_TOKEN=config["ASANA_API_TOKEN"],
    PORTFOLIO_GID=config["PORTFOLIO_GID"],
    TEAM_GID=config["TEAM_GID"]
)

# Asana API setup
configuration = asana.Configuration()
configuration.access_token = config["ASANA_API_TOKEN"]
api_client = asana.ApiClient(configuration)

# Initialize Asana API instances
portfolios_api = asana.PortfoliosApi(api_client)
projects_api = asana.ProjectsApi(api_client)
tasks_api = asana.TasksApi(api_client)
sections_api = asana.SectionsApi(api_client)


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
    return list(portfolios_api.get_items_for_portfolio(config["PORTFOLIO_GID"], opts=opts))

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

def process_tasks(tasks: List[Dict[str, Any]], project_name: str, project_gid: str) -> List[Dict[str, Any]]:
    processed_tasks = []
    for task in tasks:
        task_data = {
            'project': project_name,
            'project_gid': project_gid,  # Add this line
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
            if field_name := safe_get(field, 'name'):
                task_data[f"custom_{field_name}"] = safe_get(field, 'display_value')

        processed_tasks.append(task_data)
    return processed_tasks


def create_interactive_timeline(project_completion_estimates):
    today = pd.Timestamp.now().floor('D')
    future_placeholder_date = today + pd.DateOffset(years=2)

    # Sort projects by estimated completion date
    project_completion_estimates = project_completion_estimates.sort_values('estimated_completion_date')

    fig = go.Figure()

    # Assign a unique y value to each project to avoid overlap
    y_positions = list(range(len(project_completion_estimates)))
    project_completion_estimates['y'] = y_positions

    # Define a color map and symbols for different project statuses
    color_map = {
        'completed': '#4CAF50',  # Green
        'due_today': '#FFEB3B',  # Yellow
        'upcoming': '#FF9800',   # Orange
        'undetermined': '#F44336' # Red
    }

    symbol_map = {
        'completed': 'circle',
        'due_today': 'star',
        'upcoming': 'diamond',
        'undetermined': 'x'
    }

    # Add markers for each project
    for i, (_, project) in enumerate(project_completion_estimates.iterrows()):
        completion_date = project['estimated_completion_date']
        y_position = project['y']

        if pd.isnull(completion_date):
            status = 'undetermined'
            marker_date = future_placeholder_date
            hover_text = f"{project['project']}<br>Completion date undetermined"
        else:
            if completion_date < today:
                status = 'completed'
                hover_text = f"{project['project']}<br>Completed: {completion_date.strftime('%Y-%m-%d')}"
            elif completion_date == today:
                status = 'due_today'
                hover_text = f"{project['project']}<br>Due today: {completion_date.strftime('%Y-%m-%d')}"
            else:
                status = 'upcoming'
                hover_text = f"{project['project']}<br>Estimated completion: {completion_date.strftime('%Y-%m-%d')}"
            marker_date = completion_date

        fig.add_trace(go.Scatter(
            x=[marker_date],
            y=[y_position],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color_map[status],
                symbol=symbol_map[status],
                line=dict(width=2, color='DarkSlateGrey')  # Adding a border for better visibility
            ),
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

    # Configure layout with improved aesthetics
    fig.update_layout(
        title="Project Timeline",
        xaxis=dict(
            title="Date",
            tickformat='%Y-%m-%d',
            range=[min(today - pd.Timedelta(days=60), project_completion_estimates['estimated_completion_date'].min()),
                   max(future_placeholder_date, project_completion_estimates['estimated_completion_date'].max())],
            rangeslider=dict(visible=True),
            type="date",
            showgrid=True,  # Display grid for better readability
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title="Projects",
            tickvals=y_positions,
            ticktext=project_completion_estimates['project'],
            range=[-1, len(y_positions)],
            showgrid=False  # Keep y-axis clean without gridlines
        ),
        plot_bgcolor='white',  # Set background color to white for clarity
        height=600,
        showlegend=False,
        hovermode='closest'
    )

    return fig


def create_velocity_chart(df):
    df['completed_at'] = pd.to_datetime(df['completed_at'])
    monthly_velocity = df[df['status'] == 'Completed'].groupby(df['completed_at'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M')).size().reset_index(
        name='completed_tasks')
    monthly_velocity['completed_at'] = monthly_velocity['completed_at'].dt.to_timestamp()
    monthly_velocity['rolling_avg'] = monthly_velocity['completed_tasks'].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=monthly_velocity['completed_at'], y=monthly_velocity['completed_tasks'], name='Tasks Completed'))
    fig.add_trace(go.Scatter(x=monthly_velocity['completed_at'], y=monthly_velocity['rolling_avg'],
                             name='3-Month Rolling Average', line=dict(color='red')))

    fig.update_layout(title='Monthly Velocity Chart', xaxis_title='Month', yaxis_title='Tasks Completed')
    return fig


def create_rampup_chart(df, new_member_start_dates):
    rampup_data = []
    for member, start_date in new_member_start_dates.items():
        member_tasks = df[(df['assignee'] == member) & (df['status'] == 'Completed')]
        member_tasks['days_since_start'] = (member_tasks['completed_at'] - start_date).dt.days
        monthly_avg = member_tasks.groupby(member_tasks['days_since_start'] // 30)[
            'completion_time'].mean().reset_index()
        monthly_avg['member'] = member
        rampup_data.append(monthly_avg)

    rampup_df = pd.concat(rampup_data)

    return px.line(
        rampup_df,
        x='days_since_start',
        y='completion_time',
        color='member',
        title='New Member Ramp-up Time',
        labels={'completion_time': 'Avg Task Completion Time (days)'},
    )


def create_burndown_chart(df):
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    total_tasks = len(df)
    daily_tasks = df.groupby('date').size().cumsum()
    ideal_burndown = pd.Series(np.linspace(total_tasks, 0, len(daily_tasks)), index=daily_tasks.index)

    completed_tasks = df[df['status'] == 'Completed'].groupby('date').size().cumsum()
    remaining_tasks = total_tasks - completed_tasks

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_tasks.index, y=ideal_burndown, name='Ideal Burndown', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=remaining_tasks.index, y=remaining_tasks, name='Actual Burndown'))

    fig.update_layout(title='Project Burndown Chart', xaxis_title='Date', yaxis_title='Remaining Tasks')
    return fig


def create_cycle_time_chart(df):
    df['cycle_time'] = (df['completed_at'] - df['created_at']).dt.days
    monthly_cycle_time = df[df['status'] == 'Completed'].groupby(df['completed_at'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M'))[
        'cycle_time'].mean().reset_index()
    monthly_cycle_time['completed_at'] = monthly_cycle_time['completed_at'].dt.to_timestamp()

    return px.line(
        monthly_cycle_time,
        x='completed_at',
        y='cycle_time',
        title='Monthly Average Cycle Time',
        labels={'cycle_time': 'Avg Cycle Time (days)'},
    )


def create_cumulative_flow_diagram(df):
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    status_over_time = df.groupby(['date', 'status']).size().unstack(fill_value=0).cumsum()

    fig = go.Figure()
    for column in status_over_time.columns:
        fig.add_trace(go.Scatter(x=status_over_time.index, y=status_over_time[column],
                                 name=column, stackgroup='one', mode='none'))

    fig.update_layout(title='Cumulative Flow Diagram', xaxis_title='Date', yaxis_title='Number of Tasks')
    return fig


def get_project_owner(project_name: str, project_gid: str, projects_api: Any) -> str:
    try:
        opts = {
            'opt_fields': 'owner.name'
        }
        project_details = projects_api.get_project(project_gid, opts=opts)
        return project_details['owner']['name'] if project_details.get('owner') else "Unassigned"
    except Exception as e:
        print(f"Error fetching project owner for {project_name}: {e}")
        return "Unknown"

def get_project_members_count(project_name: str, project_gid: str, projects_api: Any) -> int:
    try:
        opts = {
            'opt_fields': 'members'
        }
        project_details = projects_api.get_project(project_gid, opts=opts)
        return len(project_details.get('members', []))
    except Exception as e:
        print(f"Error fetching project members count for {project_name}: {e}")
        return 0


def get_total_tasks(project_name: str, df: pd.DataFrame) -> int:

    return df[df['project'] == project_name].shape[0]


def get_completed_tasks(project_name: str, df: pd.DataFrame) -> int:

    return df[(df['project'] == project_name) & (df['status'] == 'Completed')].shape[0]


def get_overdue_tasks(project_name: str, df: pd.DataFrame) -> int:

    now = datetime.now(timezone.utc)
    return df[(df['project'] == project_name) &
              (df['status'] != 'Completed') &
              (df['due_date'] < now) &
              (df['due_date'].notna())].shape[0]


def get_project_gid(project_name: str, portfolios_api: Any, portfolio_gid: str) -> str:

    try:
        opts = {
            'opt_fields': 'name,gid',
        }
        portfolio_items = list(portfolios_api.get_items_for_portfolio(portfolio_gid, opts=opts))
        for item in portfolio_items:
            if item['name'] == project_name:
                return item['gid']
        raise ValueError(f"Project '{project_name}' not found in the portfolio")
    except Exception as e:
        print(f"Error fetching project GID for {project_name}: {e}")
        return "Unknown"


def get_project_details(project: Dict[str, Any], projects_api: Any, portfolios_api: Any, portfolio_gid: str, df: pd.DataFrame) -> Dict[str, Any]:
    project_name = project['project']
    project_gid = df[df['project'] == project_name]['project_gid'].iloc[0] if 'project_gid' in df.columns else None

    if project_gid is None or project_gid == "Unknown":
        return {
            'name': project_name,
            'gid': "Unknown",
            'owner': "Unknown",
            'members_count': 0,
            'total_tasks': get_total_tasks(project_name, df),
            'completed_tasks': get_completed_tasks(project_name, df),
            'overdue_tasks': get_overdue_tasks(project_name, df),
            'estimated_completion_date': project['estimated_completion_date'],
            'remaining_tasks': project['remaining_tasks'],
            'avg_task_completion_time': project['avg_task_completion_time'],
            'on_track': project['on_track'],
            'days_difference': project['days_difference']
        }
    else:
        return {
            'name': project_name,
            'gid': project_gid,
            'owner': get_project_owner(project_name, project_gid, projects_api),
            'members_count': get_project_members_count(project_name, project_gid, projects_api),
            'total_tasks': get_total_tasks(project_name, df),
            'completed_tasks': get_completed_tasks(project_name, df),
            'overdue_tasks': get_overdue_tasks(project_name, df),
            'estimated_completion_date': project['estimated_completion_date'],
            'remaining_tasks': project['remaining_tasks'],
            'avg_task_completion_time': project['avg_task_completion_time'],
            'on_track': project['on_track'],
            'days_difference': project['days_difference']
        }


def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return ['rgb({:.0f}, {:.0f}, {:.0f})'.format(x[0] * 255, x[1] * 255, x[2] * 255) for x in RGB_tuples]


def create_resource_allocation_chart(df):
    # Convert dates to datetime and remove timezone information
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)

    # Filter for the current fiscal year and remove unassigned tasks
    fiscal_year_start = pd.Timestamp('2023-10-01')
    fiscal_year_end = pd.Timestamp('2024-09-30')
    df_fiscal = df[(df['created_at'] >= fiscal_year_start) &
                   (df['created_at'] <= fiscal_year_end) &
                   (df['assignee'] != 'Unassigned')]

    # Group by assignee and month, count tasks
    monthly_allocation = df_fiscal.groupby([df_fiscal['created_at'].dt.to_period('M'), 'assignee']).size().unstack(
        fill_value=0)
    monthly_allocation.index = monthly_allocation.index.to_timestamp()

    # Generate distinct colors for each assignee
    assignees = monthly_allocation.columns
    colors = generate_distinct_colors(len(assignees))
    color_map = dict(zip(assignees, colors))

    # Create the plot
    fig = go.Figure()
    for assignee in assignees:
        fig.add_trace(go.Scatter(
            x=monthly_allocation.index,
            y=monthly_allocation[assignee],
            mode='lines+markers',
            name=assignee,
            line=dict(color=color_map[assignee])
        ))

    fig.update_layout(
        title='Resource Allocation Over Time',
        xaxis_title='Month',
        yaxis_title='Number of Tasks Assigned',
        legend_title='Team Members',
        hovermode='x unified'
    )
    return fig


def create_individual_productivity_chart(df):
    # Convert dates to datetime and ensure UTC timezone
    df['completed_at'] = pd.to_datetime(df['completed_at'], utc=True)

    # Filter for the current fiscal year, completed tasks, and remove unassigned tasks
    fiscal_year_start = pd.Timestamp('2023-10-01', tz='UTC')
    fiscal_year_end = pd.Timestamp('2024-09-30', tz='UTC')
    df_fiscal = df[(df['completed_at'] >= fiscal_year_start) &
                   (df['completed_at'] <= fiscal_year_end) &
                   (df['status'] == 'Completed') &
                   (df['assignee'] != 'Unassigned')]

    # Group by assignee and month, count completed tasks
    monthly_productivity = df_fiscal.groupby([
        df_fiscal['completed_at'].dt.to_period('M'),
        'assignee'
    ]).size().unstack(fill_value=0)

    monthly_productivity.index = monthly_productivity.index.to_timestamp()

    # Generate distinct colors for each assignee
    assignees = monthly_productivity.columns
    colors = generate_distinct_colors(len(assignees))
    color_map = dict(zip(assignees, colors))

    # Create the plot
    fig = go.Figure()
    for assignee in assignees:
        fig.add_trace(go.Scatter(
            x=monthly_productivity.index,
            y=monthly_productivity[assignee],
            mode='lines+markers',
            name=assignee,
            line=dict(color=color_map[assignee])
        ))

    fig.update_layout(
        title="Individual Productivity Over Time",
        xaxis_title="Month",
        yaxis_title="Number of Tasks Completed",
        legend_title="Team Members",
        hovermode='x unified'
    )

    return fig


# Function to create a card for each project
def create_project_card(project: Dict[str, Any], projects_api: Any, portfolios_api: Any, portfolio_gid: str,
                        df: pd.DataFrame) -> str:
    project_details = get_project_details(project, projects_api, portfolios_api, portfolio_gid, df)

    on_track = project_details['on_track']
    status_class = "status-on-track" if on_track else "status-off-track"
    status_text = "On Track" if on_track else "Off Track"

    asana_url = f"https://app.asana.com/0/{project_details['gid']}/list"

    card_html = f"""
    <div class='project-card'>
        <div class='project-title'><a href="{asana_url}" target="_blank">{project_details['name']}</a></div>
        <div class='project-status {status_class}'>{status_text}</div>
        <div class='project-details'>
            <div class='project-detail'>
                <span class='detail-label'>Estimated completion:</span><br>
                <span class='project-metric'>{project_details['estimated_completion_date'].strftime('%Y-%m-%d') if pd.notnull(project_details['estimated_completion_date']) else 'Not available'}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Remaining tasks:</span><br>
                <span class='project-metric'>{project_details['remaining_tasks']}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Overdue tasks:</span><br>
                <span class='project-metric overdue'>{project_details['overdue_tasks']}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Avg task time:</span><br>
                <span class='project-metric'>{f"{project_details['avg_task_completion_time'].days:.1f} days" if pd.notnull(project_details['avg_task_completion_time']) else 'N/A'}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Project Owner:</span><br>
                <span class='project-metric'>{project_details['owner']}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Team Members:</span><br>
                <span class='project-metric'>{project_details['members_count']}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Total Tasks:</span><br>
                <span class='project-metric'>{project_details['total_tasks']}</span>
            </div>
            <div class='project-detail'>
                <span class='detail-label'>Completed Tasks:</span><br>
                <span class='project-metric'>{project_details['completed_tasks']}</span>
            </div>
        </div>
    """
    if project_details['days_difference'] is not None:
        status = "ahead of" if project_details['days_difference'] < 0 else "behind"
        card_html += f"""
        <div class='project-detail' style='margin-top: 0.8rem;'>
            <span class='detail-label'>Schedule:</span>
            <span class='project-metric'>{abs(project_details['days_difference'])} days {status}</span>
        </div>
        """
    card_html += "</div>"
    return card_html


def calculate_resource_utilization(df):
    # Assuming 'assignee' represents resources and we're looking at active tasks
    active_tasks = df[df['status'] != 'Completed']
    resource_utilization = active_tasks['assignee'].value_counts()
    total_resources = df['assignee'].nunique()

    # Calculate utilization percentage (assuming max capacity is 10 tasks per resource)
    utilization_percentage = (len(active_tasks) / (total_resources * 10)) * 100

    return {
        'utilization_percentage': min(utilization_percentage, 100),  # Cap at 100%
        'top_utilized_resources': resource_utilization.head(5).to_dict()
    }


def get_recent_activity(df, days=7):
    recent_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)

    recent_completed = df[(df['completed_at'] >= recent_date) & (df['status'] == 'Completed')]
    recent_created = df[df['created_at'] >= recent_date]

    return {
        'completed_tasks': len(recent_completed),
        'created_tasks': len(recent_created)
    }


def calculate_time_to_completion_trend(df, months=4):
    df['completion_time'] = (df['completed_at'] - df['created_at']).dt.total_seconds() / 86400  # in days

    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - pd.DateOffset(months=months - 1)  # 3 months back + current month

    # Create a date range for each month
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    weekly_avg_completion_time = []

    for date in date_range:
        month_end = date + pd.offsets.MonthEnd(1)
        month_data = df[
            (df['completed_at'] >= date) &
            (df['completed_at'] <= month_end) &
            (df['status'] == 'Completed')
            ]
        avg_time = month_data['completion_time'].mean()
        weekly_avg_completion_time.append({
            'date': date,
            'days_to_complete': avg_time if pd.notna(avg_time) else 0
        })

    trend_df = pd.DataFrame(weekly_avg_completion_time)

    # Add a projection for the next month
    last_avg = trend_df['days_to_complete'].iloc[-1]
    next_month = end_date + pd.DateOffset(months=1)
    projection_df = pd.DataFrame([{
        'date': next_month,
        'days_to_complete': last_avg  # Simple projection using last month's average
    }])

    # Use concat instead of append
    trend_df = pd.concat([trend_df, projection_df], ignore_index=True)

    return trend_df


# Apply custom CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Refresh button
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Main content
try:
    with st.spinner("Loading portfolio projects..."):
        portfolio_projects = get_portfolio_projects()

    if portfolio_projects:
        selected_projects = [project['name'] for project in portfolio_projects]

        all_tasks = []
        for project in portfolio_projects:
            project_name = project['name']
            project_gid = project['gid']
            with st.spinner(f"Loading tasks for project: {project_name}"):
                tasks = get_tasks(project_gid)
            if tasks:
                all_tasks.extend(process_tasks(tasks, project_name, project_gid))  # Note the added project_gid here
            else:
                st.warning(f"No tasks found for project: {project_name}")

        if all_tasks:
            df = pd.DataFrame(all_tasks)
            df['due_date'] = pd.to_datetime(df['due_date'], utc=True)
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
            df['completed_at'] = pd.to_datetime(df['completed_at'], utc=True)
            project_completion_estimates = estimate_project_completion(df)

        # Define fiscal year end date
        fiscal_year_end = pd.Timestamp('2024-09-30')

        # Add on-track status column
        project_completion_estimates['on_track'] = project_completion_estimates['estimated_completion_date'].apply(
            lambda x: x <= fiscal_year_end if pd.notnull(x) else False
        )

        # Calculate all the stats
        total_projects = len(project_completion_estimates)
        on_track_projects = sum(project_completion_estimates['on_track'])
        total_remaining_tasks = project_completion_estimates['remaining_tasks'].sum()
        avg_completion_time = project_completion_estimates['avg_task_completion_time'].mean()
        total_tasks = len(df)
        completed_tasks = df['status'].value_counts().get('Completed', 0)
        completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        # New calculations
        resource_util = calculate_resource_utilization(df)
        recent_activity = get_recent_activity(df)
        completion_trend = calculate_time_to_completion_trend(df)

        # Calculate average time to complete for current month
        current_month = pd.Timestamp.now().month
        df['completion_time'] = (df['completed_at'] - df['created_at']).dt.total_seconds() / 86400  # Convert to days
        avg_completion_month = df[(df['status'] == 'Completed') & (df['completed_at'].dt.month == current_month)][
            'completion_time'].mean()

        # Calculate average time to complete for current fiscal year
        fiscal_year_start = pd.Timestamp('2023-10-01', tz='UTC')
        avg_completion_year = df[(df['status'] == 'Completed') & (df['completed_at'] >= fiscal_year_start)][
            'completion_time'].mean()

        # Convert averages to integers to truncate decimals
        avg_completion_month = int(avg_completion_month) if not pd.isna(avg_completion_month) else 0
        avg_completion_year = int(avg_completion_year) if not pd.isna(avg_completion_year) else 0

        # Convert completion_trend to DataFrame and name the column
        if isinstance(completion_trend, list):
            completion_trend = pd.DataFrame(completion_trend, columns=['days_to_complete'])

        # Generate an index or dates for the x-axis if not already provided
        completion_trend['date'] = pd.date_range(start='2023-01-01', periods=len(completion_trend), freq='W')

        # Check for missing trends and set defaults
        recent_completed_tasks_trend = recent_activity.get('completed_tasks_trend', 0)
        recent_created_tasks_trend = recent_activity.get('created_tasks_trend', 0)

        # First Row: Project Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Projects", total_projects, help="Total number of projects in the portfolio.")
        with col2:
            on_track_percentage = (on_track_projects / total_projects) * 100 if total_projects > 0 else 0
            st.metric("On Track Projects", f"{on_track_projects}/{total_projects}", f"{round(on_track_percentage, 1)}%",
                      help="Percentage of projects estimated to complete on or before the fiscal year end.")
        with col3:
            st.metric("Completion Rate", f"{completion_rate:.1f}%",
                      help="Percentage of tasks completed out of total tasks.")
        with col4:
            st.metric("Resource Utilization", f"{resource_util['utilization_percentage']:.1f}%",
                      help="Percentage of total available resources currently utilized.")

        # Second Row: Task and Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tasks", total_tasks, help="Total number of tasks across all projects.")
        with col2:
            st.metric("Completed Tasks", completed_tasks, help="Total number of completed tasks.")
        with col3:
            st.metric("Total Remaining Tasks", total_remaining_tasks,
                      help="Total number of remaining tasks to be completed.")
        with col4:
            avg_completion_days = avg_completion_time.days if pd.notnull(avg_completion_time) else 0
            st.metric("Avg Task Completion", f"{avg_completion_days:.1f} days",
                      help="Average number of days taken to complete tasks normalized across all projects.")

        # Third Row: Recent Activity and New Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recent Completed Tasks", recent_activity['completed_tasks'], f"{recent_completed_tasks_trend}%",
                      help="Number of tasks completed in the last 7 days.")
        with col2:
            st.metric("Recent Created Tasks", recent_activity['created_tasks'], f"{recent_created_tasks_trend}%",
                      help="Number of tasks created in the last 7 days.")
        with col3:
            st.metric("Avg Time to Complete (This Month)", f"{avg_completion_month} days",
                      help="Average time to complete a task for the current month.")
        with col4:
            st.metric("Avg Time to Complete (This Fiscal Year)", f"{avg_completion_year} days",
                      help="Average time to complete a task for the current fiscal year.")

        # Trend Chart: Time to Completion with Hover Data
        st.subheader("Time to Completion Trend")

        # Calculate the trend data
        completion_trend = calculate_time_to_completion_trend(df)

        # Create the trend chart with the available data
        completion_trend['color'] = completion_trend['days_to_complete'].apply(lambda x: 'green' if x < 7 else 'red')

        trend_chart = go.Figure(data=go.Scatter(
            x=completion_trend['date'],
            y=completion_trend['days_to_complete'],
            mode='lines+markers',
            marker=dict(color=completion_trend['color']),
            line=dict(color='#4299e1', width=2),
            text=completion_trend['days_to_complete'].apply(lambda x: f"<b>{x:.2f} Days</b>"),
            hoverinfo='text+x'
        ))

        # Generate tick values and labels
        tick_values = completion_trend['date']
        tick_labels = [date.strftime('%b %Y') for date in completion_trend['date']]

        trend_chart.update_layout(
            title="Time to Completion Trend",
            xaxis_title="Date",
            yaxis_title="Days to Completion",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, zeroline=False),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_labels,
                tickangle=-45
            )
        )

        st.plotly_chart(trend_chart, use_container_width=True)

        # Top Utilized Resources
        st.subheader("Top Utilized Resources")
        top_resources = list(resource_util['top_utilized_resources'].items())[:5]
        col1, col2, col3 = st.columns(3)
        for index, (resource, count) in enumerate(top_resources):
            if index % 3 == 0:
                with col1:
                    st.metric(resource, count, help=f"{count} tasks assigned to {resource}.")
            elif index % 3 == 1:
                with col2:
                    st.metric(resource, count, help=f"{count} tasks assigned to {resource}.")
            else:
                with col3:
                    st.metric(resource, count, help=f"{count} tasks assigned to {resource}.")

        # Task Status Distribution Pie Charts Side by Side
        st.subheader("Task Distribution Overview")
        col1, col2 = st.columns(2)

        with col1:
            try:
                fig_status = px.pie(df, names='status', title="Task Status Distribution",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False  # Remove top legend
                )
                st.plotly_chart(fig_status, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering Task Status Distribution: {e}")

        with col2:
            try:
                # Calculate the distribution of tasks by assignee, including 'Unassigned'
                task_assignment = df['assignee'].fillna('Unassigned').value_counts().reset_index()
                task_assignment.columns = ['Assignee', 'Task Count']

                # Create the pie chart for task assignment distribution
                fig_assignment = px.pie(task_assignment, names='Assignee', values='Task Count',
                                        title="Task Assignment Distribution",
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_assignment.update_traces(textposition='inside', textinfo='percent+label')
                fig_assignment.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False  # Remove top legend
                )
                st.plotly_chart(fig_assignment, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering Task Assignment Distribution: {e}")

        # Add Interactive Timeline after the pie charts
        st.subheader("Project Completion Timeline Overview")

        try:
            # Assuming you have a DataFrame `project_completion_estimates` ready for the timeline
            project_completion_estimates = estimate_project_completion(df)  # Or however you generate this DataFrame
            fig_timeline = create_interactive_timeline(project_completion_estimates)
            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Project Completion Timeline: {e}")

        # Custom CSS for styling
        st.markdown("""
        <style>
            .project-card {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            .project-card:hover {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }
            .project-title {
                font-size: 1.2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
                color: #2c3e50;
            }
            .project-title a {
                color: #2c3e50;
                text-decoration: none;
                transition: color 0.3s ease;
            }
            .project-title a:hover {
                color: #3498db;
            }
            .project-status {
                font-size: 0.9rem;
                font-weight: bold;
                padding: 0.3rem 0.7rem;
                border-radius: 20px;
                display: inline-block;
                margin-bottom: 0.8rem;
            }
            .status-on-track {
                background-color: #27ae60;
                color: white;
            }
            .status-off-track {
                background-color: #e74c3c;
                color: white;
            }
            .project-details {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 0.5rem;
            }
            .project-detail {
                font-size: 0.9rem;
                color: #34495e;
            }
            .detail-label {
                font-weight: bold;
                color: #7f8c8d;
            }
            .project-metric {
                font-weight: bold;
                color: #2980b9;
            }
            .project-metric.overdue {
                color: #e74c3c;
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


            # Create a grid layout for project cards (3 columns)
            for i in range(0, len(project_completion_estimates), 3):
                col1, col2, col3 = st.columns(3, gap="small")
                with col1:
                    st.markdown(create_project_card(project_completion_estimates.iloc[i], projects_api, portfolios_api,
                                                    config["PORTFOLIO_GID"], df), unsafe_allow_html=True)
                if i + 1 < len(project_completion_estimates):
                    with col2:
                        st.markdown(
                            create_project_card(project_completion_estimates.iloc[i + 1], projects_api, portfolios_api,
                                                config["PORTFOLIO_GID"], df), unsafe_allow_html=True)
                if i + 2 < len(project_completion_estimates):
                    with col3:
                        st.markdown(
                            create_project_card(project_completion_estimates.iloc[i + 2], projects_api, portfolios_api,
                                                config["PORTFOLIO_GID"], df), unsafe_allow_html=True)

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
                avg_completion_time.append(0 if pd.isna(avg_time) else avg_time)

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
                    'avg_completion_time': 'N/A' if pd.isna(avg_time) else f"{avg_time:.2f}",
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
        st.subheader("Velocity Chart")
        velocity_chart = create_velocity_chart(df)
        st.plotly_chart(velocity_chart)

        st.subheader("Burndown Chart")
        burndown_chart = create_burndown_chart(df)
        st.plotly_chart(burndown_chart)

        st.subheader("Cycle Time Analysis")
        cycle_time_chart = create_cycle_time_chart(df)
        st.plotly_chart(cycle_time_chart)

        st.subheader("Cumulative Flow Diagram")
        cfd_chart = create_cumulative_flow_diagram(df)
        st.plotly_chart(cfd_chart)

        st.subheader("Resource Allocation Over Time")
        resource_allocation_chart = create_resource_allocation_chart(df)
        st.plotly_chart(resource_allocation_chart)

        st.subheader("Individual Productivity Over Time")
        individual_productivity_chart = create_individual_productivity_chart(df)
        st.plotly_chart(individual_productivity_chart)
    else:
        st.info("No projects found in the portfolio.")

# Run the app with: streamlit run asana_dashboard.py
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.exception(e)

if st.sidebar.button("Clear Saved Credentials"):
    for key in ["ASANA_API_TOKEN", "PORTFOLIO_GID", "TEAM_GID"]:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("Credentials cleared. Please refresh the page.")
    st.rerun()
