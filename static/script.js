document.getElementById("metricsForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    const response = await fetch("/update-metrics", {
        method: "POST",
        body: formData
    });
    const result = await response.json();
    alert(result.message);

    // Update displayed metrics without refreshing
    document.getElementById("accuracy").textContent = formData.get("accuracy");
    document.getElementById("precision").textContent = formData.get("precision");
    document.getElementById("recall").textContent = formData.get("recall");
});
