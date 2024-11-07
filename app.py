from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "5nhPfssLpK"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1,1),Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "plot1.png"
    # Replace with code to generate and save the scatter plot
    plt.figure()
    plt.scatter(X, Y, label='Data')
    plt.plot(X, model.predict(X.reshape(-1,1)), color='red', label='Fitted Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"assign7/laimao19-assignment-7/static/{plot1_path}")
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1,1), Y_sim)
        sim_slope = sim_model.coef_[0] 
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "plot2.png"
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label='Slopes')
    plt.hist(intercepts, bins=20, alpha=0.5, label='Intercepts')
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.legend()
    plt.savefig(f"assign7/laimao19-assignment-7/static/{plot2_path}")
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(np.abs(slopes) >= np.abs(slope)) / S  
    intercept_extreme = sum(np.abs(intercepts) >= np.abs(intercept)) / S  

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
        parameter_label = "Slope"
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0
        parameter_label = "Intercept"

    # TODO 10: Calculate p-value based on test type
    if test_type == ">":
        p_value = sum(simulated_stats >= observed_stat) / S
    elif test_type == "<":
        p_value = sum(simulated_stats <= observed_stat) / S
    elif test_type == "!=":
        p_value = sum(np.abs(simulated_stats) >= np.abs(observed_stat)) / S

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = "You've encountered a rare event!" if p_value <= 0.0001 else ""

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=20, color='skyblue', alpha=0.6, label="Simulated Statistics")
    plt.axvline(observed_stat, color='red', linestyle="--", label=f"Observed {parameter_label}: {observed_stat:.4f}")
    plt.axvline(hypothesized_value, color='blue', linestyle="-", label=f"Hypothesized {parameter_label} (H₀): {hypothesized_value}")
    plt.xlabel(parameter_label)
    plt.ylabel("Frequency")
    plt.title(f"Hypothesis Test for {parameter_label}")
    plt.legend()
    plt.savefig(f"assign7/laimao19-assignment-7/static/{plot3_path}")
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="plot1.png",
        plot2="plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        # TODO 13: Uncomment the following lines when implemented
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))
    confidence_level = confidence_level / 100 if confidence_level > 1 else confidence_level

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
        parameter_label = "Slope"
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0
        parameter_label = "Intercept"

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    if std_estimate == 0:
        # If std_estimate is zero, confidence interval is just the mean
        ci_lower, ci_upper = mean_estimate, mean_estimate
    else:
        # Calculate t-value for the specified confidence level
        t_value = t.ppf((1 + confidence_level) / 2, len(estimates) - 1)
        margin = t_value * (std_estimate / np.sqrt(len(estimates)))
        ci_lower, ci_upper = mean_estimate - margin, mean_estimate + margin


    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "plot4.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(estimates, [0] * len(estimates), color='gray', alpha=0.5, label="Simulated Estimates")
    plt.plot([mean_estimate], [0], 'bo', label="Mean Estimate")
    plt.hlines(0, ci_lower, ci_upper, colors='blue', linewidth=3, label=f"{confidence_level*100:.1f}% Confidence Interval")
    plt.axvline(true_param, color='green', linestyle="-.", label="True Parameter")

    plt.xlabel(f"{parameter_label} Estimate")
    plt.yticks([])  
    plt.title(f"{confidence_level*100:.1f}% Confidence Interval for {parameter_label} (Mean Estimate)")
    plt.legend()
    plt.savefig(f"assign7/laimao19-assignment-7/static/{plot4_path}")
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="plot1.png",
        plot2="plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
