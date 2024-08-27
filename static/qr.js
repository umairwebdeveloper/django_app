(function () {
	// Scoping function to avoid globals
	var isMobile = /(Mobile|Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini)/i.test(navigator.userAgent) || false;
	var button = document.getElementById('scan-button');
	var text = document.getElementById('qr-text');
	var qrCode = document.getElementById('qr-code');
	button.style.display = 'none';
	text.style.display = 'none';
	qrCode.style.display = 'none';
	if (!isMobile) {
		text.style.display = 'block';
		qrCode.style.display = 'block';
		text.style.marginBottom = '1rem';
		new QRCode(qrCode, {
			text: 'https://cssscript.com',
			width: 250,
			height: 250,
			logo: 'https://api.shoefitr.io/static/logo.png',
			logoWidth: 65,
			logoHeight: 41,
			colorDark: '#3865f3',
			typeNumber: 5
		});
	} else if (isMobile) {
		button.type = 'button';
		button.style.display = 'block';
		button.addEventListener('click', function () {
			document.getElementById('shoefitr-web').style.display = 'block';
			button.style.display = 'none';
		});
	}
})();
