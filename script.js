document.getElementById("predictForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const humidity = document.getElementById("humidity").value;
    const windSpeed = document.getElementById("windSpeed").value;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ humidity: humidity, wind_speed: windSpeed })
        });

        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            throw new Error("Server did not return JSON.");
        }

        const data = await response.json();
        console.log("Server Response:", data);

        if (data.error) {
            document.getElementById("result").innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        } else {
            document.getElementById("result").innerHTML = `
                <p>Predicted Temperature: ${data["Predicted Temperature (°C)"] || "N/A"}°C</p>
                <p>Predicted Weather: ${data["Predicted Weather Condition"] || "N/A"}</p>
            `;
        }
    } catch (error) {
        console.error("Fetch error:", error);
        document.getElementById("result").innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    }
});
