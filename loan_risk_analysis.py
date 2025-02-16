#%%%
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from multiprocessing import Pool

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

# Sidebar for Inputs
st.sidebar.header("Input Parameters")

# Input Widgets
with st.sidebar:
    N = st.number_input("Number of Loans, N", min_value=1, value=10_000)
    loan = st.number_input("Loan Amount, Y ($)", min_value=0, value=180_000)
    loss = st.number_input("Loss per Foreclosure, L ($)", min_value=0, value=200_000)
    r = st.slider("Interest Rate, r (%)", min_value=0.0, max_value=10.0, value=2.4, step=0.1) / 100
    p = st.slider("Probability of Default, p (%)", min_value=0.0, max_value=6.0, value=2.0, step=0.01) / 100
    P_loss_des = st.slider("Desired Probability of Losing Money (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.01) / 100


    st.write("Monte Carlo Simulation:")
    paths = st.slider("Number of paths for Monte Carlo Simulation", min_value=1000, max_value=10_000, value=5000, step=100)
    p_ud = st.slider("Probability of Default Changing", min_value=0.0, max_value=100.0, value=0.0, step=0.01) / 100
    def_ud = st.slider("Percentage change in default amount if it changes (%)", min_value=0.0, max_value=2.0, value=1.0, step=0.01) / 100
    p_up = st.slider("Probability of Default Changing Up if it changes", min_value=0.0, max_value=100.0, value=50.0, step=0.01) / 100


# Streamlit App Title
st.title("Loan Risk Analysis Dashboard")

# Display the Plot
st.write(
    """
    This dashboard tells the story of the reasons behind the big short crisis in 2008.
    Firstly, let's consider you're a bank and you have a loan portfolio.
    The loan portfolio consists of $N$ loans with a given amount ($Y$) for each loan.
    The probability of default for each loan is $p$; meaning that there is a probability of $p$ that the loan will not be paid back.
    The loss per foreclosure is also given, $L$ ($L > Y$). 
    To compensate for the risk, you charge an interest rate $r$ for each loan.
    The expected value of the earnings from one loan is given by the formula: 
    
    $$\\mathbb{E}(X) = (1-p) \\times  r \\times Y - p \\times L$$

    for a total of N loans, the sum of the earnings is given by the formula:
    
    $$S_N = X_1 + X_2 + \\ldots + X_N = \\sum_{i=1}^{N} X_i$$

    The expected value of the total earnings is given by the formula:

    $$\\mathbb{E}(S_N) = N \\times \\mathbb{E}(X)$$

    The standard error of the total earnings is given by the formula:

    $$SE(S_N) = \\sqrt{(r \\times Y - L)^2 \\times p \\times (1-p) \\times N}$$

    When the expected value of the sum of the earnings and the standard error are known, the probability of losing money can be calculated using the normal distribution.
    Here it is assumed that the total earnings are normally distributed, which is true based on the Central Limit Theorem.
    So the probability of losing money is given by the formula:

    $$ P(\\text{earning} < 0) = \\Phi\\left(\\frac{0 - \\mathbb{E}(S_N)}{SE(S_N)}\\right)$$

    where $\\Phi$ is the cumulative distribution function of the standard normal distribution.

    Set the parameters for Loan Amount ($Y$), Loss per Foreclosure ($L$), Interest Rate ($r$), Probability of Default ($p$) 
    for the loan portfolio in the sidebar and see the distribution of earnings according to the CLT. 
    Also results of the Monte Carlo simulation are shown for the same parameters.
    """)

#% Calculate Probability of Losing Money
P_loss, EX, SE = probability_of_losing_money(N, p, r, loss, loan)
x = np.linspace(EX - 4 * SE, EX + 4 * SE, 1000)
x_conf = np.linspace(EX - 2 * SE, EX + 2 * SE, 1000)
pdf = norm.pdf(x, loc=EX, scale=SE)
pdf_conf = norm.pdf(x_conf, loc=EX, scale=SE)

#% Monte Carlo simulation
outcomes = np.array([monte_carlo_simulation(N, p, r, loss, loan, p_ud, def_ud, p_up) for _ in range(paths)])


#% Plotting section
fig_pdf = go.Figure()
fig_pdf.add_trace(go.Scatter(
    x=x, 
    y=pdf, 
    mode='lines', 
    name='PDF',
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
    fillcolor='rgba(0,100,80,0.5)', 
    name='95% Confidence',
))
fig_pdf.add_vline(
    x=0, 
    line_dash="dash", 
    line_color="white", 
    name="Break-Even point",
)
fig_pdf.add_trace(go.Histogram(
    x=outcomes,
    histnorm='probability density',
    name='MC Simulation',
))
fig_pdf.update_layout(
    title='Probability Density Function', 
    xaxis_title='Total Earnings ($)', 
    yaxis_title='Density',
    # legend=dict(
    #     yanchor="top",
    #     y=1.3,
    #     xanchor="left",
    #     x=0.0,
    # ),
)

st.plotly_chart(fig_pdf)

st.write("""
    The next figure shows the probability of losing money for both the ideal case (CLT) and the Monte Carlo simulation.
    You should see that the probability of losing money is the same for both cases when the default settings are used.
    """)

# probability of losing money for the ideal case and Monte Carlo simulation
P_loss_ideal = norm.cdf(0, loc=EX, scale=SE)
P_loss_MC = np.sum(outcomes < 0) / paths

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

st.plotly_chart(fig_compare)

st.write(
    """
    Next, it is important to set the interest rate ($r$) so that the probability of losing money is below a certain threshold.
    As a first step it would be good to find the interest rate for breaking even.

    $$ r_{\\text{break-even}} = \\frac{p\\times L}{(1-p)\\times Y} $$
    """)
st.write(f"The break-even interest rate for the current settings is {p * loss / ((1 - p) * loan):.2%}.")
st.write(
    """
    Let's say that you want the probability of bank losing money due to the loan portfolio to be less than 1%.
    The interest rate according to this desired probability can be calculated using the same formula as before.
    But this time, the interest rate is the variable to be found.
    The analytical solution is quite complex, since the probability of losing money is a quadratic function of the interest rate.
    But you can change the interest rate and see the probability of losing money for different interest rates.
    This dashboard calculates the interest rate for the desired probability of losing money using the analytical solution.
    Figure below shows how changing the interest rate around the optimal value affects the probability of losing money.
    """)

#% Interest Rate Range (for the probability of losing money vs interest rate plot)
r_des_anal = find_interest_rate_for_target_loss_probability(N, p, loss, loan, P_loss_des)
r_arr = np.linspace(r_des_anal-0.001, r_des_anal+0.001, 1000)

# Calculate Probability of Losing Money
P_loss, _, _ = probability_of_losing_money(N, p, r_arr, loss, loan)

# calculate the interest rate according to the desired probability of losing money
r_des = r_arr[np.argmin(np.abs(P_loss - P_loss_des))]  # desired interest rate

fig_r = go.Figure()
fig_r.add_trace(go.Scatter(
    x=r_arr,
    y=P_loss,
    mode='lines',
    name='Ideal case (CLT)',
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
    xaxis_title='Interest Rate (%)',
    yaxis_title='Probability of Losing Money (%)',
    # x axis is percentage (show one decimal)
    xaxis_tickformat=',.2%',
    yaxis_tickformat=',.2%',
)

st.write(f"For the current settings, the interest rate that corresponds to the desired probability of losing money ({P_loss_des:.2%}) is {r_des:.2%}:")
st.plotly_chart(fig_r)

st.write(
    """
    Lastly, effect of the changes in the probability of default on the total earnings is assessed using the Monte Carlo simulation.
    One of the fundamental assumptions of the probability theory is that the probability of one event is independent of the other events.
    However, this assumption may not always be true in real life scenarios.
    For instance, the probability of default for all loans in a bank's portfolio may change due to a global economic crisis.
    In this case, the probability of default for all loans may increase (or decrease).
    In this last section, the probability of default for all loans is assumed to change with a certain probability.
    The percentage change in the probability of default is also assumed to be given.
    The probability of default can change up or down with a another certain probability.
    You can change these parameters (as well as the number of paths for the Monte Carlo simulation) and see how the total earnings change.
    As an example, you can set the probability of default changing to 50% (i.e. there is a 50% probability that the default rate changes for all laons), and 
    the percentage change in the default rate to +0.4%. Also set the probability of default changing up to 50%. 
    You can see how the probability of losing money increases significantly in this case (MC simulations).
    This is a very simplified example of how the global economic crisis can affect the banks' loan portfolios.
    """)