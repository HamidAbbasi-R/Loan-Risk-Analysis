#%%%
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# Function to calculate probability of losing money
def probability_of_losing_money(N, p, r, loss, loan):
    win = loan * r
    EX = N * (1 - p) * win - N * p * loss  # expected value
    SE = np.sqrt((win - loss) ** 2 * p * (1 - p) * N)  # standard error
    P_loss = norm.cdf(0, loc=EX, scale=SE)  # probability of losing money
    return P_loss, EX, SE

def monte_carlo_simulation(N, p, r, loss, loan, p_ud, def_ud, p_up):
    # calculate the new probability of default
    changing = np.random.choice([1, 0], p=[p_ud, 1 - p_ud])     # 1: change, 0: no change (for everyone)
    updown_ud = np.random.choice([1, -1], p=[p_up, 1 - p_up])   # 1: change up, -1: change down
    changing *= updown_ud
    p += changing * def_ud

    # Calculate the win amount
    win = loan * r

    # Simulate outcomes of N loans
    outcomes = np.random.choice([win, -loss], size=N, p=[1 - p, p])
    
    # Calculate total net outcome
    total_outcome = np.sum(outcomes)
    
    return total_outcome

def find_interest_rate_for_target_loss_probability(N, p, loss, loan, P_loss_des):
    # Z-score for the desired probability
    z_score = norm.ppf(P_loss_des)
    
    # Coefficients for the quadratic equation
    A = (1 - p)**2 * loan**2 - (z_score**2 / N) * p * (1 - p) * loan**2
    B = -2 * p * loss * (1 - p) * loan + (z_score**2 / N) * 2 * p * (1 - p) * loan * loss
    C = p**2 * loss**2 - (z_score**2 / N) * p * (1 - p) * loss**2
    
    # Solve the quadratic equation
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        raise ValueError("No real solution for the interest rate.")
    
    r1 = (-B + np.sqrt(discriminant)) / (2 * A)
    r2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    # Choose the positive root
    r = max(r1, r2) if r1 > 0 and r2 > 0 else (r1 if r1 > 0 else r2)
    
    return r

# Streamlit App Title
st.title("Loan Risk Analysis Dashboard")

# Sidebar for Inputs
st.sidebar.header("Input Parameters")

# Input Widgets
with st.sidebar:
    # add text here
    st.write("Parameters for the loan risk analysis:")
    N = st.number_input("Number of Loans (N)", min_value=1, value=10_000)
    loan = st.number_input("Loan Amount", min_value=0, value=180_000)
    loss = st.number_input("Loss per Foreclosure", min_value=0, value=200_000)
    r = st.slider("Interest Rate (r%)", min_value=0.0, max_value=10.0, value=2.4, step=0.1) / 100
    p = st.slider("Probability of Default (%)", min_value=0.0, max_value=6.0, value=2.0, step=0.01) / 100
    P_loss_des = st.slider("Desired Probability of Losing Money (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.01) / 100

    st.write("Monte Carlo Simulation:")
    paths = st.slider("Number of Paths", min_value=1000, max_value=50_000, value=5000, step=100)
    p_ud = st.slider("Probability of Default Changing (or Otherwise Remaining the Same)", min_value=0.0, max_value=100.0, value=0.0, step=0.01) / 100
    st.write(f"(Interpretations: There is a probability of {p_ud:.2%} that the default rate of all loans changes.)")
    def_ud = st.slider("Percentage change in default amount if it changes (%)", min_value=0.0, max_value=2.0, value=1.0, step=0.01) / 100
    p_up = st.slider("Probability of Default Changing Up if it changes", min_value=0.0, max_value=100.0, value=50.0, step=0.01) / 100
    MC_paths_for_r_des = st.slider("Number of Paths for Interest Rate Calculation (Note: intense calculations in case paths>>1000)", min_value=100, max_value=5_000, value=100, step=100)

#% Calculate Probability of Losing Money
P_loss, EX, SE = probability_of_losing_money(N, p, r, loss, loan)
x = np.linspace(EX - 4 * SE, EX + 4 * SE, 1000)
x_conf = np.linspace(EX - 2 * SE, EX + 2 * SE, 1000)
pdf = norm.pdf(x, loc=EX, scale=SE)
pdf_conf = norm.pdf(x_conf, loc=EX, scale=SE)
P_loss_ideal = norm.cdf(0, loc=EX, scale=SE)

#% Monte Carlo simulation
outcomes = np.array([monte_carlo_simulation(N, p, r, loss, loan, p_ud, def_ud, p_up) for _ in range(paths)])
P_loss_MC = np.sum(outcomes < 0) / paths


#% Interest Rate Range (for the probability of losing money vs interest rate plot)
r_des_anal = find_interest_rate_for_target_loss_probability(N, p, loss, loan, P_loss_des)
r_arr = np.linspace(r_des_anal-0.001, r_des_anal+0.001, 1000)

# Calculate Probability of Losing Money
P_loss, _, _ = probability_of_losing_money(N, p, r_arr, loss, loan)

# Monte Carlo simulation for different interest rates
r_arr_MC = np.linspace(r_des_anal-0.001, r_des_anal+0.001, 50)
P_loss_MC_r_des = np.zeros_like(r_arr_MC)
for i, r in enumerate(r_arr_MC):
    outcomes_r_des = np.array([monte_carlo_simulation(N, p, r, loss, loan, p_ud, def_ud, p_up) for _ in range(MC_paths_for_r_des)])
    P_loss_MC_r_des[i] = np.sum(outcomes_r_des < 0) / MC_paths_for_r_des



# calculate the interest rate according to the desired probability of losing money
r_des = r_arr[np.argmin(np.abs(P_loss - P_loss_des))]  # desired interest rate


#% Plotting section
fig_pdf = go.Figure()
fig_pdf.add_trace(go.Scatter(
    x=x, 
    y=pdf, 
    mode='lines', 
    name='PDF (Normal Distribution) for Ideal Case',
))
fig_pdf.add_vline(
    x=EX, 
    line_dash="dash", 
    line_color="red", 
    annotation_text="Expected Value", 
    annotation_position="bottom right",
)
fig_pdf.add_trace(go.Scatter(
    x=x_conf, 
    y=pdf_conf, 
    fill='tozeroy', 
    mode='none', 
    fillcolor='rgba(0,100,80,0.2)', 
    name='95% Confidence Interval',
))
fig_pdf.add_vline(
    x=0, 
    line_dash="dash", 
    line_color="white", 
    name="Break-Even point",
)
fig_pdf.add_histogram(
    x=outcomes, 
    histnorm='probability density',     # options are 'probability', 'density', 'probability density'
    name='Monte Carlo Simulation', 
    nbinsx = 50,
)
fig_pdf.update_layout(
    title='Probability Density Function', 
    xaxis_title='Total Earnings ($)', 
    yaxis_title='Density',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.7,
    ),
)

# compare the P_loss_ideal and P_loss_MC
fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(
    x=['Ideal Case', 'Monte Carlo Simulation'],
    y=[P_loss_ideal, P_loss_MC],
    name='Probability of Losing Money',
    text=[f'{P_loss_ideal:.2%}', f'{P_loss_MC:.2%}'],
))
fig_compare.update_layout(
    title='Comparison of Probability of Losing Money',
    yaxis_title='Probability of Losing Money',
    barmode='group',
    # y axis is percentage (show one decimal)
    yaxis_tickformat=',.1%',
)

fig_r = go.Figure()
fig_r.add_trace(go.Scatter(
    x=r_arr,
    y=P_loss,
    mode='lines',
    name='Ideal case',
))
fig_r.add_trace(go.Scatter(
    x=r_arr_MC,
    y=P_loss_MC_r_des,
    mode='lines',
    name='MC Simulation',
))
fig_r.add_trace(go.Scatter(
    x=r_arr,
    y= P_loss_des * np.ones_like(r_arr),
    mode='lines',
    # dash line
    line_dash='dash',
    name=f'P(Loss) = {P_loss_des:.2%}',
))
fig_r.update_layout(
    title='Probability of Losing Money',
    xaxis_title='Interest Rate',
    yaxis_title='Probability of Losing Money',
    # x axis is percentage (show one decimal)
    xaxis_tickformat=',.1%',
)


# Display the Plot
st.write(
    """
    This dashboard simulates the risk of losing money on a given loan portfolio. 
    Besides the typical calculations that are based on probability theory, it uses Monte-Carlo simulation for a more realistic approach. 
    In the MC simulation, there could be a non-zero probability that default rate of all loans changes. 
    Then there is another probability that the default rate changes up or down. 
    The percentage change in the default rate is also considered.
    """)
st.write(f"For the interest rate of {r:.2%}, distribution of the earnings is as follows:")
st.plotly_chart(fig_pdf)

st.write(
    """
    The next figure shows how much differnce is observed between the ideal case and the Monte Carlo simulation.
    The bar chart shows the probability of losing money for both cases. 
    It is shown that for a MC simulation that does not allow the default rate to change, the probability of losing money is the same as the ideal case. 
    """)
st.plotly_chart(fig_compare)

st.write(
    """
    The next figure shows the probability of losing money for different interest rates.
    For a given desired probability of losing money, the corresponding interest rate can be found using this plot.
    """)
st.write(f"The interest rate that corresponds to the desired probability of losing money ({P_loss_des:.2%}) is {r_des:.2%}:")
st.plotly_chart(fig_r)