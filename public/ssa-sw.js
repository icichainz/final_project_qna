const CACHE_NAME = "ssa-chatbot-brand-v1";
const BRAND_ASSETS = [
  "/favicon",
  "/public/logo_light.png",
  "/public/logo_dark.png",
  "/public/avatars/ssa_chatbot.png",
  "/public/brand/icons/icon-192.png",
  "/public/brand/icons/icon-512.png",
  "/public/brand/icons/icon-maskable-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(BRAND_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key.startsWith("ssa-chatbot-") && key !== CACHE_NAME)
        .map((key) => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const isBrandAsset = url.origin === self.location.origin &&
    (url.pathname === "/favicon" || url.pathname.startsWith("/public/brand/") ||
     url.pathname.startsWith("/public/avatars/") || url.pathname.startsWith("/public/logo_"));
  if (!isBrandAsset) return;
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
});
