'use strict';

/**
 * register-image service
 */

const { createCoreService } = require('@strapi/strapi').factories;

module.exports = createCoreService('api::register-image.register-image');
