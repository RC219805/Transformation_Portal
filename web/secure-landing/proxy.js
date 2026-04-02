import { NextResponse } from "next/server.js";

const SESSION_COOKIE_NAME = "__Host-tp_session";

export function proxy(request) {
  const hasSessionCookie = Boolean(request.cookies.get(SESSION_COOKIE_NAME)?.value);
  const { pathname } = request.nextUrl;

  if (pathname === "/login" && hasSessionCookie) {
    return NextResponse.redirect(new URL("/portal", request.url), 302);
  }

  const response = NextResponse.next();
  response.headers.set("Referrer-Policy", "same-origin");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("X-Content-Type-Options", "nosniff");
  return response;
}

export const config = {
  matcher: ["/login"]
};
