function googleTranslateElementInit() {
  new google.translate.TranslateElement(
    {
      pageLanguage: "en",
      includedLanguages: "en,nl,es,fr,de,zh-CN,it,pt",
      layout: google.translate.TranslateElement.InlineLayout.VERTICAL,
    },
    "google_translate_element"
  );
}

var prevScrollpos = window.pageYOffset;
window.onscroll = function () {
  try {
    var currentScrollPos = window.pageYOffset;
    if (prevScrollpos > currentScrollPos) {
      document.getElementById("shoefitr-navbar").style.top = "0";
    } else {
      document.getElementById("shoefitr-navbar").style.top = "-80px";
    }
    prevScrollpos = currentScrollPos;
    console.log("scrolling");
  } catch (e) {
    console.log(e);
  }
};

// Wait for element to be added to the DOM
function waitForElm(selector) {
  return new Promise((resolve) => {
    if (document.querySelector(selector)) {
      return resolve(document.querySelector(selector));
    }

    const observer = new MutationObserver((mutations) => {
      if (document.querySelector(selector)) {
        resolve(document.querySelector(selector));
        observer.disconnect();
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  });
}

// Remove the google powered by Logo
waitForElm(".goog-te-gadget").then((elem) => {
  if (elem) {
    // Remove all children except the first one
    while (elem.childNodes.length > 1) {
      elem.removeChild(elem.lastChild);
    }
  }
});

