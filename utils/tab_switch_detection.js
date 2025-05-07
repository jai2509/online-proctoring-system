document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
        alert("You have switched tabs. This activity is being monitored.");
        // Log this event to the backend if needed
    }
});
