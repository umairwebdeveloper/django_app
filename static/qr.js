document.addEventListener("DOMContentLoaded", function () {
	// Function to hide elements initially
	const hideElements = (elements) => {
		elements.forEach((el) => {
			if (el) el.style.display = "none";
		});
	};

	// Function to display QR code and text for non-mobile devices
	const displayQRCode = (qrCode, text) => {
		if (qrCode && text) {
			text.style.display = "block";
			text.style.marginBottom = "1rem";
			qrCode.style.display = "block";

			const qrData = {
				text: "https://cssscript.com",
				size: "250x250",
				color: "3865f3",
			};

			const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(
				qrData.text
			)}&size=${qrData.size}&color=${qrData.color}`;

			// Insert QR code image
			qrCode.innerHTML = `<img src="${qrUrl}" alt="QR Code" onerror="this.onerror=null;this.src='/path-to-your-fallback-image.png';">`;
		}
	};

	// Function to handle mobile-specific behavior
	const handleMobile = (button, shoefitrWeb) => {
		if (button && shoefitrWeb) {
			button.type = "button";
			button.style.display = "block";

			button.addEventListener("click", function () {
				shoefitrWeb.style.display = "block";
				button.style.display = "none";
			});
		}
	};

	// Main logic
	(function () {
		const isMobile =
			/Mobile|Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
				navigator.userAgent
			);

		const button = document.getElementById("scan-button");
		const text = document.getElementById("qr-text");
		const qrCode = document.getElementById("qr-code");
		const shoefitrWeb = document.getElementById("shoefitr-web");

		// Initially hide elements
		hideElements([button, text, qrCode]);

		// Display appropriate content based on device type
		if (!isMobile) {
			displayQRCode(qrCode, text);
		} else {
			handleMobile(button, shoefitrWeb);
		}
	})();
});
