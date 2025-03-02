import { GH_TOKEN } from "./settings.js"
//const fetch = require('fetch')
//const base64 = require('base64')

function decodeFile(content) {
    return base64.b64decode(fileData).decode('utf-8')
}

async function main(){
    const url = "https://api.github.com/repos/iamkun/dayjs/contents/README.md"
    const response = await fetch(url, {method : "GET", headers : {"Authorization": `token ${GH_TOKEN}`}})

    if (response.ok) {
        const content = (await response.json()).content
        console.log(decodeFile(content))
    } else
        console.log(response.reason)
}

main()