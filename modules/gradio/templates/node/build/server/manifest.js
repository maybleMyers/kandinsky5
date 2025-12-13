const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.BZ-4Ut_E.js","app":"_app/immutable/entry/app.CdtO6O9O.js","imports":["_app/immutable/entry/start.BZ-4Ut_E.js","_app/immutable/chunks/client.Bp30c0_t.js","_app/immutable/entry/app.CdtO6O9O.js","_app/immutable/chunks/preload-helper.DpQnamwV.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-Ch43j9k-.js')),
			__memo(() => import('./chunks/1-Ddbqj5gn.js')),
			__memo(() => import('./chunks/2-CGsk4wvi.js').then(function (n) { return n.aJ; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
