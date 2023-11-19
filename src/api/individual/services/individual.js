'use strict';

/**
 * individual service
 */

const { createCoreService } = require('@strapi/strapi').factories;

module.exports = createCoreService('api::individual.individual');
