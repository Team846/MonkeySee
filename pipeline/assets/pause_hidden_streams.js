document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    document.querySelectorAll('img[src*="/video_feed/"]').forEach(function (img) {
      img.dataset.feed = img.src;
      img.src = "data:,";
    });
  } else {
    document.querySelectorAll("img[data-feed]").forEach(function (img) {
      img.src = img.dataset.feed;
      delete img.dataset.feed;
    });
  }
});
