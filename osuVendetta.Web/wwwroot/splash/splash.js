function hideSplashScreen() {
    var splashScreen = document.getElementById('splash-screen');

    if (splashScreen == null)
        return;

    splashScreen.hidden = true;
}

function getLoadPercentage() {
    var loadPercentage = getComputedStyle(document.documentElement)
        .getPropertyValue('--blazor-load-percentage');

    if (loadPercentage == null)
        return 0;

    return Number(loadPercentage.replace('%', ''));
}

function updateLoadPercentageText() {
    var textContainer = document.getElementById('load-percentage');
    var loadingBar = document.getElementById('splash-progress-bar-value');
    var loadPercentage = getLoadPercentage();

    loadingBar.style = `width: ${loadPercentage}%;`;
    textContainer.textContent = `${loadPercentage} %`;

    if (loadPercentage == 100) {
        window.clearInterval(window.loadUpdateTimer);
    }
}

if (!window.loadUpdateTimer)
    window.loadUpdateTimer = setInterval(updateLoadPercentageText, 500);