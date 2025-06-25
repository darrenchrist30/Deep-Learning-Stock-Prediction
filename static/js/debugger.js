// Debug helper for custom plot
document.addEventListener("DOMContentLoaded", function () {
  // Find the update button
  const updatePlotBtn = document.getElementById("updatePlot");

  if (updatePlotBtn) {
    // Patch the original fetch with a monitored version
    const origFetch = window.fetch;
    window.fetch = function () {
      console.log("INTERCEPTED FETCH:", arguments);

      // If this is the custom_prediction endpoint
      if (arguments[0] === "/custom_prediction") {
        console.log("CUSTOM PREDICTION REQUEST:", arguments[1].body);

        return origFetch.apply(this, arguments).then((response) => {
          // Clone the response so we can both read it and return it
          const responseClone = response.clone();

          responseClone
            .json()
            .then((data) => {
              console.log("CUSTOM PREDICTION RESPONSE:", {
                success: data.success,
                message: data.message,
                htmlLength: data.html_length,
                plotHtmlFirstChars: data.plot_html
                  ? data.plot_html.substring(0, 100) + "..."
                  : "EMPTY",
                hasPlotlyJs:
                  data.plot_html && data.plot_html.includes("Plotly.newPlot"),
                hasValidDiv: data.plot_html && data.plot_html.includes("<div"),
                fullResponse: data,
              });
            })
            .catch((err) => {
              console.error("Error parsing response JSON:", err);
            });

          return response;
        });
      }

      return origFetch.apply(this, arguments);
    };

    // Add a more thorough debugging for the custom plot container
    const originalUpdatePlot = updatePlotBtn.onclick;
    updatePlotBtn.addEventListener("click", function () {
      const customPlotContainer = document.getElementById(
        "customPlotContainer"
      );
      const defaultPlot = document.getElementById("defaultPlot");

      console.log("BEFORE REQUEST - Container states:", {
        customPlotContainerDisplay: customPlotContainer
          ? getComputedStyle(customPlotContainer).display
          : "ELEMENT_NOT_FOUND",
        defaultPlotDisplay: defaultPlot
          ? getComputedStyle(defaultPlot).display
          : "ELEMENT_NOT_FOUND",
        customPlotContainerHTML: customPlotContainer
          ? customPlotContainer.innerHTML
          : "ELEMENT_NOT_FOUND",
      });

      // Setup observer to detect DOM changes
      if (customPlotContainer) {
        const observer = new MutationObserver((mutationsList) => {
          console.log("DOM MUTATION DETECTED:", mutationsList);
          console.log("UPDATED CONTAINER:", {
            customPlotContainerDisplay:
              getComputedStyle(customPlotContainer).display,
            customPlotContainerHTML: customPlotContainer.innerHTML,
            hasPlotly:
              customPlotContainer.querySelector(".js-plotly-plot") !== null,
          });
        });

        observer.observe(customPlotContainer, {
          childList: true,
          subtree: true,
          attributes: true,
          characterData: true,
        });

        // Disconnect after 5 seconds to avoid memory leaks
        setTimeout(() => observer.disconnect(), 5000);
      }
    });
  }
});
