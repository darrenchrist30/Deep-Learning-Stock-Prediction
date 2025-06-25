// Script for Stock Price Prediction App

document.addEventListener("DOMContentLoaded", function () {
  // File input validation for CSV
  const fileInput = document.getElementById("file");
  if (fileInput) {
    fileInput.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        // Check file extension
        const extension = file.name.split(".").pop().toLowerCase();
        if (extension !== "csv") {
          alert("Please select a CSV file.");
          fileInput.value = "";
        }
      }
    });
  }

  // File input validation for H5 model
  const modelInput = document.getElementById("model_file");
  if (modelInput) {
    modelInput.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        // Check file extension
        const extension = file.name.split(".").pop().toLowerCase();
        if (extension !== "h5") {
          alert("Please select an H5 model file.");
          modelInput.value = "";
        }
      }
    });
  }

  // File input validation for PKL scaler
  const scalerInput = document.getElementById("scaler_file");
  if (scalerInput) {
    scalerInput.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        // Check file extension
        const extension = file.name.split(".").pop().toLowerCase();
        if (extension !== "pkl") {
          alert("Please select a PKL scaler file.");
          scalerInput.value = "";
        }
      }
    });
  }
  // Auto-dismiss alerts after 5 seconds
  const alerts = document.querySelectorAll(".alert");
  alerts.forEach((alert) => {
    setTimeout(() => {
      const bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    }, 5000);
  });

  // Handle upload model form submission via AJAX
  const uploadModelForm = document.getElementById("upload_model_form");
  if (uploadModelForm) {
    uploadModelForm.addEventListener("submit", function (event) {
      event.preventDefault();

      const formData = new FormData(uploadModelForm);

      fetch("/upload_model", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Create notification
          const alertBox = document.createElement("div");
          alertBox.className = `alert alert-${
            data.success ? "success" : "danger"
          } alert-dismissible fade show`;
          alertBox.innerHTML = `
          ${data.message}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

          // Add to page
          const cardBody = uploadModelForm.closest(".card-body");
          cardBody.insertBefore(alertBox, uploadModelForm);

          // Update model status
          if (data.success) {
            const modelStatusAlert =
              document.querySelector(".alert:last-child");
            if (modelStatusAlert) {
              modelStatusAlert.className = "alert alert-success";
              modelStatusAlert.querySelector("p").textContent =
                "Model is loaded and ready to use.";
            }
          }

          // Auto dismiss after 5 seconds
          setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alertBox);
            bsAlert.close();
          }, 5000);
        })
        .catch((error) => {
          console.error("Error:", error);
          alert("An error occurred while loading the model.");
        });
    });
  }
});
